"""Plot reward curves for multiple TurningDoubleWell potential configurations."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils.turning_double_well_env import TurningDoubleWellEnv


def _parse_float_list(raw: str) -> list[float]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return [float(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/turning_double_well/reward_configs_over_action.png"),
    )
    parser.add_argument(
        "--random-states-output-path",
        type=Path,
        default=Path("artifacts/turning_double_well/reward_configs_over_action__random_states.png"),
        help="Output path for the random-state subplot plot.",
    )
    parser.add_argument("--well-heights", type=str, default="0.1,0.3,0.6")
    parser.add_argument("--peak-degs", type=str, default="35,45,55")
    parser.add_argument("--max-turn-deg", type=float, default=90.0)
    parser.add_argument("--reference-heading-deg", type=float, default=0.0)
    parser.add_argument("--num-points", type=int, default=401)
    parser.add_argument(
        "--random-states",
        type=int,
        default=10,
        help="If >0, also plot the reward profile over actions for this many random starting headings.",
    )
    parser.add_argument(
        "--random-states-cols",
        type=int,
        default=None,
        help="Number of subplot columns for the random-state plot (defaults to an automatic grid).",
    )
    parser.add_argument("--random-state-seed", type=int, default=0, help="Seed for random starting headings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    heights = _parse_float_list(args.well_heights)
    peak_degs = _parse_float_list(args.peak_degs)

    actions = np.linspace(-1.0, 1.0, args.num_points, dtype=np.float32)
    delta_degrees = actions * args.max_turn_deg
    delta_radians = jnp.deg2rad(jnp.asarray(delta_degrees, dtype=jnp.float32))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for well_height, peak_deg in product(heights, peak_degs):
        env = TurningDoubleWellEnv(
            horizon=1,
            max_turn_deg=args.max_turn_deg,
            well_angle_deg=peak_deg,
            well_height=well_height,
        )
        rewards = np.asarray(env.reward_from_angle(delta_radians))
        ax.plot(
            actions,
            rewards,
            linewidth=2.0,
            label=f"h0={well_height:g}, peak={peak_deg:g}deg",
        )

    ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.8)
    ax.axhline(1.0, color="0.55", linestyle="--", linewidth=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Normalized action")
    ax.set_ylabel("Reward")
    ax.set_title("TurningDoubleWell reward for configuration sweeps")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    tick_actions = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    top_ax.set_xticks(tick_actions)
    tick_degrees = tick_actions * args.max_turn_deg
    top_ax.set_xticklabels([f"{deg:.0f}" for deg in tick_degrees])
    top_ax.set_xlabel("Turn delta [deg]")

    fig.tight_layout()
    fig.savefig(args.output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {args.output_path}")

    if int(args.random_states) > 0:
        out_path = args.random_states_output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Use one representative configuration (first in the sweep) and overlay many
        # "reward over action" curves for random initial headings.
        env = TurningDoubleWellEnv(
            horizon=1,
            max_turn_deg=args.max_turn_deg,
            well_angle_deg=float(peak_degs[0]),
            well_height=float(heights[0]),
            randomize_initial_heading=True,
            snap_action_to_optimal=True,
        )

        rng = np.random.default_rng(int(args.random_state_seed))
        headings_deg = rng.uniform(-180.0, 180.0, size=(int(args.random_states),)).astype(np.float32)

        rewards = np.asarray(env.reward_from_angle(delta_radians))

        n = int(args.random_states)
        cols = (
            int(args.random_states_cols)
            if args.random_states_cols is not None
            else int(np.ceil(np.sqrt(n)))
        )
        cols = max(1, cols)
        rows = int(np.ceil(n / cols))

        fig2, axes2 = plt.subplots(
            nrows=rows,
            ncols=cols,
            figsize=(3.6 * cols, 2.8 * rows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes2_flat = axes2.reshape(-1)
        for idx, heading in enumerate(headings_deg):
            ax = axes2_flat[idx]
            ax.plot(actions, rewards, color="tab:red", linewidth=2.0)
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"θ0={float(heading):.0f}°", fontsize=9)

        for ax in axes2_flat[n:]:
            ax.axis("off")

        for ax in axes2[-1, :]:
            ax.set_xlabel("Normalized action")
        for ax in axes2[:, 0]:
            ax.set_ylabel("Reward")

        fig2.suptitle(
            "TurningDoubleWell reward over action for random headings "
            f"(peak={peak_degs[0]:g}deg, h0={heights[0]:g}, max_turn={args.max_turn_deg:g}deg)\n"
            "Note: reward depends only on the turn delta, so all subplots should match.",
            fontsize=11,
        )

        fig2.tight_layout(rect=(0, 0, 1, 0.93))
        fig2.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved random-state plot to {out_path}")


if __name__ == "__main__":
    main()
