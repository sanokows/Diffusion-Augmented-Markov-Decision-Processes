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
    parser.add_argument("--well-heights", type=str, default="0.1,0.3,0.6")
    parser.add_argument("--peak-degs", type=str, default="35,45,55")
    parser.add_argument("--max-turn-deg", type=float, default=90.0)
    parser.add_argument("--reference-heading-deg", type=float, default=0.0)
    parser.add_argument("--num-points", type=int, default=401)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    heights = _parse_float_list(args.well_heights)
    peak_degs = _parse_float_list(args.peak_degs)

    actions = np.linspace(-1.0, 1.0, args.num_points, dtype=np.float32)
    heading_degrees = args.reference_heading_deg + actions * args.max_turn_deg
    heading_radians = jnp.deg2rad(jnp.asarray(heading_degrees, dtype=jnp.float32))

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
        rewards = np.asarray(env.reward_from_angle(heading_radians))
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
    tick_degrees = args.reference_heading_deg + tick_actions * args.max_turn_deg
    top_ax.set_xticklabels([f"{deg:.0f}" for deg in tick_degrees])
    top_ax.set_xlabel("Heading [deg]")

    fig.tight_layout()
    fig.savefig(args.output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {args.output_path}")


if __name__ == "__main__":
    main()
