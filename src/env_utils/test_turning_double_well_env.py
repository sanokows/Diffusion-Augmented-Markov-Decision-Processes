"""Smoke-test and visualize the TurningDoubleWellEnv."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils.turning_double_well_env import TurningDoubleWellEnv


# python /home/it4i-sanokows/code/DIMEReppo/src/env_utils/test_turning_double_well_env.py \
#   --output-dir /home/it4i-sanokows/code/DIMEReppo/artifacts/turning_double_well_batch10 \
#   --num-trajectories 10 \
#   --horizon 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/turning_double_well"))
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transition-noise-deg", type=float, default=5)
    parser.add_argument("--well-height", type=float, default=0.3)
    parser.add_argument("--well-angle-deg", type=float, default=45.0)
    parser.add_argument("--gif-fps", type=int, default=8)
    return parser.parse_args()


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def constant_negative_peak_action(env: TurningDoubleWellEnv, batch_size: int) -> np.ndarray:
    action = -env.well_angle_deg / env.max_turn_deg
    action = np.clip(action, -1.0, 1.0)
    return np.full((batch_size,), action, dtype=np.float32)


def random_band_action(
    env: TurningDoubleWellEnv,
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    sampled_deg = rng.uniform(-env.well_angle_deg, env.well_angle_deg, size=(batch_size,))
    return np.asarray(sampled_deg / env.max_turn_deg, dtype=np.float32)


def rollout_policy_batch(
    env: TurningDoubleWellEnv,
    *,
    batch_size: int,
    seed: int,
    policy_name: str,
) -> tuple[list, np.ndarray]:
    key = jax.random.PRNGKey(seed)
    batch_keys = jax.random.split(key, batch_size)
    state = env.reset(batch_keys)
    states = [state]
    rewards = []
    rng = np.random.default_rng(seed)

    for _ in range(env.horizon):
        if policy_name == "policy_a":
            action_value = constant_negative_peak_action(env, batch_size)
        elif policy_name == "policy_b":
            action_value = random_band_action(env, rng, batch_size)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")
        action = jnp.asarray(action_value[:, None], dtype=jnp.float32)
        state = env.step(state, action)
        states.append(state)
        rewards.append(np.asarray(state.reward, dtype=np.float32))
        if bool(np.asarray(jnp.all(state.done))):
            break

    return states, np.asarray(rewards, dtype=np.float32)


def smoke_test(env: TurningDoubleWellEnv) -> None:
    key = jax.random.PRNGKey(123)
    masked_env = TurningDoubleWellEnv(mask_state_in_observation=True)
    masked_state = masked_env.reset(key)
    assert masked_state.obs.shape == (2,)
    assert np.allclose(np.asarray(masked_state.obs), 0.0, atol=1e-6)

    unmasked_env = TurningDoubleWellEnv(mask_state_in_observation=False)
    state = unmasked_env.reset(key)
    assert state.obs.shape == (2,)
    assert np.isclose(np.linalg.norm(np.asarray(state.obs)), 1.0, atol=1e-5)

    randomized_env = TurningDoubleWellEnv(
        mask_state_in_observation=False,
        randomize_initial_heading=True,
    )
    rand_state_a = randomized_env.reset(jax.random.PRNGKey(0))
    rand_state_b = randomized_env.reset(jax.random.PRNGKey(1))
    rand_state_a2 = randomized_env.reset(jax.random.PRNGKey(0))
    assert not np.allclose(np.asarray(rand_state_a.direction), np.asarray(rand_state_b.direction))
    assert np.allclose(np.asarray(rand_state_a.direction), np.asarray(rand_state_a2.direction))
    assert not np.array_equal(np.asarray(rand_state_a.rng), np.asarray(jax.random.PRNGKey(0)))

    next_state = unmasked_env.step(state, jnp.asarray([0.0], dtype=jnp.float32))
    assert next_state.obs.shape == (2,)
    assert np.isclose(np.linalg.norm(np.asarray(next_state.obs)), 1.0, atol=1e-5)
    assert not bool(np.asarray(next_state.done))

    batch_keys = jax.random.split(key, 4)
    batch_state = unmasked_env.reset(batch_keys)
    batch_action = jnp.zeros((4, 1), dtype=jnp.float32)
    batch_next = unmasked_env.step(batch_state, batch_action)
    assert batch_next.obs.shape == (4, 2)
    assert batch_next.reward.shape == (4,)

    state = unmasked_env.reset(key)
    for _ in range(unmasked_env.horizon):
        state = unmasked_env.step(state, jnp.asarray([0.0], dtype=jnp.float32))
    assert bool(np.asarray(state.done))
    assert np.isclose(float(np.asarray(state.info["truncation"])), 1.0)

    # Auto-reset on the next step after termination.
    reset_state = unmasked_env.step(state, jnp.asarray([0.0], dtype=jnp.float32))
    assert not bool(np.asarray(reset_state.done))
    assert int(np.asarray(reset_state.t)) == 0
    assert np.isclose(float(np.asarray(reset_state.info["truncation"])), 0.0)

    sample_angles = jnp.linspace(-jnp.pi, jnp.pi, 2049, dtype=jnp.float32)
    sample_rewards = np.asarray(unmasked_env.reward_from_angle(sample_angles))
    assert np.all(sample_rewards <= 1.0 + 1e-5)
    assert np.all(sample_rewards >= -1.0 - 1e-5)

    anchor_degrees = np.asarray(
        [-90.0, 0.0, -unmasked_env.well_angle_deg, unmasked_env.well_angle_deg, 90.0],
        dtype=np.float32,
    )
    anchor_radians = jnp.deg2rad(jnp.asarray(anchor_degrees))
    anchor_rewards = np.asarray(unmasked_env.reward_from_angle(anchor_radians))
    assert np.isclose(anchor_rewards[0], 0.0, atol=1e-4)
    assert np.isclose(anchor_rewards[1], unmasked_env.well_height, atol=1e-4)
    assert np.isclose(anchor_rewards[2], 1.0, atol=1e-4)
    assert np.isclose(anchor_rewards[3], 1.0, atol=1e-4)
    assert np.isclose(anchor_rewards[4], 0.0, atol=1e-4)

    pos_tail_deg = np.linspace(unmasked_env.well_angle_deg, 90.0, 300, dtype=np.float32)
    pos_tail_rewards = np.asarray(unmasked_env.reward_from_angle(jnp.deg2rad(jnp.asarray(pos_tail_deg))))
    assert np.all(np.diff(pos_tail_rewards) <= 1e-6)

    neg_tail_deg = np.linspace(-90.0, -unmasked_env.well_angle_deg, 300, dtype=np.float32)
    neg_tail_rewards = np.asarray(unmasked_env.reward_from_angle(jnp.deg2rad(jnp.asarray(neg_tail_deg))))
    assert np.all(np.diff(neg_tail_rewards) >= -1e-6)

    low_center_env = TurningDoubleWellEnv(well_height=0.15)
    high_center_env = TurningDoubleWellEnv(well_height=0.55)
    sample_angle = jnp.deg2rad(jnp.asarray(22.5, dtype=jnp.float32))
    low_center_reward = float(np.asarray(low_center_env.reward_from_angle(sample_angle)))
    high_center_reward = float(np.asarray(high_center_env.reward_from_angle(sample_angle)))
    assert not np.isclose(low_center_reward, high_center_reward)


def plot_reward_landscape(env: TurningDoubleWellEnv, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees, potential, reward = env.reward_landscape()
    peak_deg = env.well_angle_deg
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(degrees, potential, color="tab:red", linewidth=2.0)
    axes[0].axvline(-peak_deg, color="tab:green", linestyle="--", alpha=0.7)
    axes[0].axvline(peak_deg, color="tab:green", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Potential")
    axes[0].set_title("TurningDoubleWellEnv reward landscape")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(degrees, reward, color="tab:blue", linewidth=2.0)
    axes[1].axvline(-peak_deg, color="tab:green", linestyle="--", alpha=0.7)
    axes[1].axvline(peak_deg, color="tab:green", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Heading angle [deg]")
    axes[1].set_ylabel("Reward")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_action_potential(env: TurningDoubleWellEnv, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actions, heading_degrees, potential = env.action_potential_profile()
    peak_action = np.clip(env.well_angle_deg / env.max_turn_deg, -1.0, 1.0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(actions, potential, color="tab:red", linewidth=2.0)
    ax.axvline(-peak_action, color="tab:green", linestyle="--", alpha=0.7)
    ax.axvline(peak_action, color="tab:green", linestyle="--", alpha=0.7)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Normalized action")
    ax.set_ylabel("Potential")
    ax.set_title("Potential over one-step actions")
    ax.grid(True, alpha=0.3)

    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    tick_actions = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    top_ax.set_xticks(tick_actions)
    top_ax.set_xticklabels([f"{deg:.0f}" for deg in np.linspace(heading_degrees[0], heading_degrees[-1], 5)])
    top_ax.set_xlabel("Resulting heading [deg] from heading 0")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_policy_comparison(
    env: TurningDoubleWellEnv,
    policy_a_states: list,
    policy_b_states: list,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    preferred = np.asarray(
        [
            [np.cos(float(env.well_angle_radians)), np.sin(float(env.well_angle_radians))],
            [np.cos(float(env.well_angle_radians)), -np.sin(float(env.well_angle_radians))],
        ],
        dtype=np.float32,
    )

    policy_a_positions = env.rollout_positions(policy_a_states)
    policy_b_positions = env.rollout_positions(policy_b_states)

    for ax, positions_batch, title in zip(
        axes,
        [policy_a_positions, policy_b_positions],
        [
            f"Policy A: constant -{env.well_angle_deg:g} deg relative action",
            f"Policy B: random action in [-{env.well_angle_deg:g}, {env.well_angle_deg:g}] deg",
        ],
    ):
        for vec in preferred:
            ax.plot(
                [0.0, env.horizon * vec[0]],
                [0.0, env.horizon * vec[1]],
                "--",
                color="tab:green",
                linewidth=1.0,
                alpha=0.4,
            )
        for rollout_idx in range(positions_batch.shape[1]):
            positions = positions_batch[:, rollout_idx]
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                linewidth=1.8,
                alpha=0.8,
            )
            ax.scatter(positions[-1, 0], positions[-1, 1], s=24)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")

    axes[0].set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_comparison_gif(
    env: TurningDoubleWellEnv,
    policy_a_states: list,
    policy_a_rewards: np.ndarray,
    policy_b_states: list,
    policy_b_rewards: np.ndarray,
    output_path: Path,
    fps: int,
) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("imageio is required to save the comparison GIF.") from exc

    positions_a = env.rollout_positions(policy_a_states)
    positions_b = env.rollout_positions(policy_b_states)
    titles_a = [f"A{i}" for i in range(positions_a.shape[1])]
    titles_b = [f"B{i}" for i in range(positions_b.shape[1])]

    frames_a = env.render_trajectory(
        positions_a,
        rewards=policy_a_rewards,
        overlay=False,
        width=240,
        height=220,
        titles=titles_a,
        max_envs=positions_a.shape[1],
    )
    frames_b = env.render_trajectory(
        positions_b,
        rewards=policy_b_rewards,
        overlay=False,
        width=240,
        height=220,
        titles=titles_b,
        max_envs=positions_b.shape[1],
    )
    combined_frames = [
        np.concatenate([frame_a, frame_b], axis=1)
        for frame_a, frame_b in zip(frames_a, frames_b)
    ]
    imageio.mimsave(output_path, combined_frames, fps=fps)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = TurningDoubleWellEnv(
        horizon=args.horizon,
        well_height=args.well_height,
        well_angle_deg=args.well_angle_deg,
        transition_noise_deg=args.transition_noise_deg,
    )
    smoke_test(env)

    plot_action_potential(env, args.output_dir / "action_potential.png")
    policy_a_states, policy_a_rewards = rollout_policy_batch(
        env,
        batch_size=args.num_trajectories,
        seed=args.seed,
        policy_name="policy_a",
    )
    policy_b_states, policy_b_rewards = rollout_policy_batch(
        env,
        batch_size=args.num_trajectories,
        seed=args.seed + 10_000,
        policy_name="policy_b",
    )

    plot_reward_landscape(env, args.output_dir / "reward_landscape.png")
    plot_policy_comparison(
        env,
        policy_a_states,
        policy_b_states,
        args.output_dir / "policy_comparison.png",
    )
    save_comparison_gif(
        env,
        policy_a_states,
        policy_a_rewards,
        policy_b_states,
        policy_b_rewards,
        args.output_dir / "policy_comparison.gif",
        args.gif_fps,
    )

    print(f"Saved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
