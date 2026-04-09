"""Plot per-start-heading reward landscapes for turning environments.

This script renders one figure for `TurningDoubleWellEnv` and one for
`TurningMultiWellEnv`. Each subplot corresponds to one unique starting heading
from the environment's snapped reset support.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils.turning_double_well_env import TurningDoubleWellEnv
from src.env_utils.turning_multi_well_env import TurningMultiWellEnv


def wrap_degrees(angle_deg: np.ndarray | float) -> np.ndarray:
    angle_deg = np.asarray(angle_deg, dtype=np.float32)
    return ((angle_deg + 180.0) % 360.0) - 180.0


def unique_start_headings_deg(env: object) -> np.ndarray:
    support = getattr(env, "_snapped_initial_heading_support_radians", None)
    if support is None:
        heading = np.rad2deg(np.asarray(getattr(env, "initial_heading_radians"), dtype=np.float32))
        support_deg = np.asarray([heading], dtype=np.float32)
    else:
        support_deg = np.rad2deg(np.asarray(support, dtype=np.float32))
    wrapped = wrap_degrees(support_deg)
    return np.unique(np.round(wrapped, decimals=6)).astype(np.float32)


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
        return np.asarray([], dtype=np.float32)

    split_points = np.where(np.diff(candidate_idx) > 1)[0] + 1
    clusters = np.split(candidate_idx, split_points)

    maxima = []
    for cluster in clusters:
        best = cluster[np.argmax(reward[cluster])]
        maxima.append(float(degrees[best]))

    maxima = np.asarray(maxima, dtype=np.float32)
    return np.unique(np.round(maxima, decimals=6)).astype(np.float32)


@dataclass(frozen=True)
class EnvPlotSpec:
    name: str
    env: object
    output_filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/turning_reward_landscapes"),
        help="Directory where per-environment figures are saved.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1001,
        help="Resolution for reward curves over actions.",
    )
    parser.add_argument(
        "--global-max-atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance when extracting global reward maxima.",
    )
    parser.add_argument(
        "--subplot-cols",
        type=int,
        default=None,
        help="Number of columns for the start-heading subplot grid.",
    )

    parser.add_argument("--max-turn-deg", type=float, default=90.0)

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


def build_plot_specs(args: argparse.Namespace) -> list[EnvPlotSpec]:
    common = {
        "horizon": 1,
        "max_turn_deg": args.max_turn_deg,
        "randomize_initial_heading": True,
        "snap_action_to_optimal": True,
    }

    double_env = TurningDoubleWellEnv(
        **common,
        well_angle_deg=args.double_well_angle_deg,
        well_height=args.double_well_height,
    )
    multi_env = TurningMultiWellEnv(
        **common,
        num_minima=args.multi_num_minima,
        well_height=args.multi_well_height,
        multi_minima_tail_power=args.multi_tail_power,
        include_opposite_headings=args.multi_include_opposite_headings,
    )

    return [
        EnvPlotSpec(
            name="TurningDoubleWellEnv",
            env=double_env,
            output_filename="double_well_reward_landscape_by_start_heading.png",
        ),
        EnvPlotSpec(
            name="TurningMultiWellEnv",
            env=multi_env,
            output_filename="multi_well_reward_landscape_by_start_heading.png",
        ),
    ]


def plot_env_reward_landscapes(
    spec: EnvPlotSpec,
    *,
    output_dir: Path,
    num_points: int,
    global_max_atol: float,
    subplot_cols: int | None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headings_deg = unique_start_headings_deg(spec.env)
    actions = np.linspace(-1.0, 1.0, num_points, dtype=np.float32)
    turn_deg = actions * float(spec.env.max_turn_deg)
    reward = np.asarray(spec.env.reward_from_angle(jnp.deg2rad(jnp.asarray(turn_deg, dtype=jnp.float32))))

    maxima_turn_deg = global_reward_maxima_turns_deg(
        spec.env,
        num_points=max(2001, num_points),
        atol=global_max_atol,
    )

    n_plots = int(headings_deg.size)
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

    for idx, start_heading in enumerate(headings_deg):
        ax = axes_flat[idx]
        ax.plot(
            turn_deg,
            reward,
            color="tab:blue",
            linewidth=1.8,
        )

        for maxima in maxima_turn_deg:
            ax.axvline(float(maxima), color="tab:green", linestyle="--", linewidth=1.0, alpha=0.7)

        ax.set_xlim(-float(spec.env.max_turn_deg), float(spec.env.max_turn_deg))
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"start heading = {float(start_heading):.1f} deg", fontsize=10)

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Relative turn angle wrt agent orientation [deg]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Reward")

    maxima_str = ", ".join(f"{float(v):.2f}" for v in maxima_turn_deg)
    fig.suptitle(
        f"{spec.name}: reward landscape for each unique start heading\n"
        f"Global reward maxima in relative turn space [deg]: [{maxima_str}]",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    output_path = output_dir / spec.output_filename
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_plot_specs(args)
    for spec in specs:
        output_path = plot_env_reward_landscapes(
            spec,
            output_dir=args.output_dir,
            num_points=args.num_points,
            global_max_atol=args.global_max_atol,
            subplot_cols=args.subplot_cols,
        )
        print(f"Saved {spec.name} figure to {output_path}")


if __name__ == "__main__":
    main()
