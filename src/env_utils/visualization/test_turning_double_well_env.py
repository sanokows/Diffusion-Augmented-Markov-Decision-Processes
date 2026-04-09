"""Simulate turning-env agents that randomly sample global reward maxima actions.

For each selected environment, this script:
1. Finds global reward maxima from `reward_landscape`.
2. Runs batched rollouts where each step samples uniformly from those maxima.
3. Saves reward-landscape and rollout summary plots (plus optional GIF).
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
from src.env_utils.turning_multi_well_env import TurningMultiWellEnv


@dataclass(frozen=True)
class EnvRunSpec:
    slug: str
    name: str
    env: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        choices=["double", "multi", "both"],
        default="both",
        help="Which environment(s) to simulate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/turning_minima_sampling"),
    )
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--num-trajectories", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turn-deg", type=float, default=90.0)
    parser.add_argument("--transition-noise-deg", type=float, default=0.0)
    parser.add_argument("--landscape-points", type=int, default=4001)
    parser.add_argument("--global-max-atol", type=float, default=1e-4)
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument(
        "--save-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a trajectory GIF per environment.",
    )

    parser.add_argument("--double-well-angle-deg", type=float, default=45.0)
    parser.add_argument("--double-well-height", type=float, default=0.3)

    parser.add_argument("--multi-num-minima", type=int, default=4)
    parser.add_argument("--multi-well-height", type=float, default=0.3)
    parser.add_argument("--multi-tail-power", type=float, default=2.0)
    parser.add_argument(
        "--multi-include-opposite-headings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def build_env_specs(args: argparse.Namespace) -> list[EnvRunSpec]:
    common = {
        "horizon": args.horizon,
        "max_turn_deg": args.max_turn_deg,
        "transition_noise_deg": args.transition_noise_deg,
        "randomize_initial_heading": True,
        "snap_action_to_optimal": False,
    }

    specs: list[EnvRunSpec] = []

    if args.env in ("double", "both"):
        specs.append(
            EnvRunSpec(
                slug="double_well",
                name="TurningDoubleWellEnv",
                env=TurningDoubleWellEnv(
                    **common,
                    well_angle_deg=args.double_well_angle_deg,
                    well_height=args.double_well_height,
                ),
            )
        )

    if args.env in ("multi", "both"):
        specs.append(
            EnvRunSpec(
                slug="multi_well",
                name="TurningMultiWellEnv",
                env=TurningMultiWellEnv(
                    **common,
                    num_minima=args.multi_num_minima,
                    well_height=args.multi_well_height,
                    multi_minima_tail_power=args.multi_tail_power,
                    include_opposite_headings=args.multi_include_opposite_headings,
                ),
            )
        )

    return specs


def global_reward_maxima_turns_deg(
    env: object,
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


def minima_actions_from_turns(env: object, turn_deg: np.ndarray) -> np.ndarray:
    actions = np.asarray(turn_deg / float(env.max_turn_deg), dtype=np.float32)
    return np.clip(actions, -1.0, 1.0)


def rollout_random_global_maxima_policy(
    env: object,
    *,
    seed: int,
    batch_size: int,
    maxima_actions: np.ndarray,
) -> tuple[list, np.ndarray, np.ndarray]:
    key = jax.random.PRNGKey(seed)
    reset_keys = jax.random.split(key, batch_size)
    state = env.reset(reset_keys)

    states = [state]
    rewards = []
    sampled_actions = []

    rng = np.random.default_rng(seed)
    for _ in range(env.horizon):
        selected_idx = rng.integers(0, maxima_actions.size, size=(batch_size,))
        action_values = maxima_actions[selected_idx]

        action = jnp.asarray(action_values[:, None], dtype=jnp.float32)
        state = env.step(state, action)

        states.append(state)
        rewards.append(np.asarray(state.reward, dtype=np.float32))
        sampled_actions.append(np.asarray(action_values, dtype=np.float32))

        if bool(np.asarray(jnp.all(state.done))):
            break

    return (
        states,
        np.asarray(rewards, dtype=np.float32),
        np.asarray(sampled_actions, dtype=np.float32),
    )


def plot_reward_landscape_with_maxima(
    env: object,
    *,
    maxima_turn_deg: np.ndarray,
    output_path: Path,
    num_points: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees, potential, reward = env.reward_landscape(num_points=num_points)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(degrees, potential, color="tab:red", linewidth=2.0)
    axes[0].set_ylabel("Potential")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(degrees, reward, color="tab:blue", linewidth=2.0)
    for peak in maxima_turn_deg:
        axes[1].axvline(float(peak), color="tab:green", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].set_xlabel("Relative turn angle [deg]")
    axes[1].set_ylabel("Reward")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    peak_text = ", ".join(f"{float(v):.2f}" for v in maxima_turn_deg)
    fig.suptitle(f"Reward landscape with global maxima at [{peak_text}] deg")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_summary(
    env: object,
    *,
    states: list,
    rewards: np.ndarray,
    sampled_actions: np.ndarray,
    maxima_turn_deg: np.ndarray,
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
    ax_traj.set_title("Agent trajectories")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.25)

    ax_actions = axes[1]
    ax_actions.hist(sampled_turn_deg.reshape(-1), bins=40, color="tab:purple", alpha=0.85)
    for peak in maxima_turn_deg:
        ax_actions.axvline(float(peak), color="tab:green", linestyle="--", linewidth=1.0)
    ax_actions.set_title("Sampled turn angles")
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
    fig.suptitle(f"Random global-maxima policy | mean return = {mean_return:.3f} +- {std_return:.3f}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_rollout_gif(
    env: object,
    *,
    states: list,
    rewards: np.ndarray,
    output_path: Path,
    fps: int,
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
    )
    imageio.mimsave(output_path, frames, fps=fps)


def run_env_simulation(spec: EnvRunSpec, args: argparse.Namespace, run_seed: int) -> None:
    env_dir = args.output_dir / spec.slug
    env_dir.mkdir(parents=True, exist_ok=True)

    maxima_turn_deg = global_reward_maxima_turns_deg(
        spec.env,
        num_points=args.landscape_points,
        atol=args.global_max_atol,
    )
    maxima_actions = minima_actions_from_turns(spec.env, maxima_turn_deg)
    if maxima_actions.size == 0:
        raise RuntimeError(f"No global maxima actions found for {spec.name}.")

    states, rewards, sampled_actions = rollout_random_global_maxima_policy(
        spec.env,
        seed=run_seed,
        batch_size=args.num_trajectories,
        maxima_actions=maxima_actions,
    )

    plot_reward_landscape_with_maxima(
        spec.env,
        maxima_turn_deg=maxima_turn_deg,
        output_path=env_dir / "reward_landscape_with_global_maxima.png",
        num_points=args.landscape_points,
    )
    plot_rollout_summary(
        spec.env,
        states=states,
        rewards=rewards,
        sampled_actions=sampled_actions,
        maxima_turn_deg=maxima_turn_deg,
        output_path=env_dir / "random_global_maxima_policy_summary.png",
    )

    if args.save_gif:
        try:
            save_rollout_gif(
                spec.env,
                states=states,
                rewards=rewards,
                output_path=env_dir / "random_global_maxima_policy.gif",
                fps=args.gif_fps,
            )
        except RuntimeError as exc:
            print(f"Skipped GIF for {spec.name}: {exc}")

    peak_text = ", ".join(f"{float(v):.3f}" for v in maxima_turn_deg)
    print(f"[{spec.name}] global maxima turn angles [deg]: {peak_text}")
    print(f"[{spec.name}] saved artifacts to {env_dir}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_env_specs(args)
    for env_idx, spec in enumerate(specs):
        run_env_simulation(spec, args, run_seed=args.seed + 10_000 * env_idx)


if __name__ == "__main__":
    main()
