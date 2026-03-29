"""Compute TurningDoubleWell rewards over an action range.

By default this evaluates actions in [-1.1, 1.1] and prints:
- normalized action
- mapped turn angle in degrees
- reward
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils.turning_double_well_env import TurningDoubleWellEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-action", type=float, default=-1.1)
    parser.add_argument("--max-action", type=float, default=1.1)
    parser.add_argument("--num-points", type=int, default=221)

    parser.add_argument("--max-turn-deg", type=float, default=90.0)
    parser.add_argument("--well-angle-deg", type=float, default=45.0)
    parser.add_argument("--well-height", type=float, default=0.85)

    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to save CSV output. If omitted, prints CSV to stdout.",
    )
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> TurningDoubleWellEnv:
    return TurningDoubleWellEnv(
        horizon=1,
        max_turn_deg=args.max_turn_deg,
        randomize_initial_heading=False,
        snap_action_to_optimal=False,
        well_angle_deg=args.well_angle_deg,
        well_height=args.well_height,
    )


def main() -> None:
    args = parse_args()
    if args.num_points < 2:
        raise ValueError("num-points must be >= 2")
    if args.max_action <= args.min_action:
        raise ValueError("max-action must be > min-action")

    env = build_env(args)

    actions = np.linspace(args.min_action, args.max_action, args.num_points, dtype=np.float32)
    turn_deg = actions * float(env.max_turn_deg)
    turn_rad = np.deg2rad(turn_deg).astype(np.float32)

    rewards = np.asarray(env.reward_from_angle(jnp.asarray(turn_rad, dtype=jnp.float32)), dtype=np.float32)

    rows = np.column_stack([actions, turn_deg, rewards])
    header = "action,turn_deg,reward"

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.csv_out, rows, delimiter=",", header=header, comments="")
        print(f"Saved CSV to {args.csv_out}")
    else:
        np.savetxt(sys.stdout, rows, delimiter=",", header=header, comments="")

    min_idx = int(np.argmin(rewards))
    max_idx = int(np.argmax(rewards))
    print(
        "Summary: "
        f"min_reward={rewards[min_idx]:.6f} at action={actions[min_idx]:.6f}, "
        f"max_reward={rewards[max_idx]:.6f} at action={actions[max_idx]:.6f}"
    )


if __name__ == "__main__":
    main()
