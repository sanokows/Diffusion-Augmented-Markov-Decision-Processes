"""Load saved torso COM trajectories and render figures."""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.env_utils.torso_com import build_torso_com_traj_figure


def _load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=os.path.join("results", "trajectories"),
        help="Directory containing trajectory pickle files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save figures (default: <input-dir>/figures).",
    )
    parser.add_argument(
        "--pattern",
        default="*_traj.pkl",
        help="Glob pattern for trajectory pickle files.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(input_dir, args.pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No trajectory pickle files found in {input_dir} (pattern={args.pattern})."
        )

    for path in paths:
        payload = _load_pickle(path)
        torso_com_traj = payload.get("torso_com_traj")
        torso_com_env_indices = payload.get("torso_com_env_indices")
        fig = build_torso_com_traj_figure(
            torso_com_traj, torso_com_env_indices, title=os.path.basename(path)
        )
        if fig is None:
            continue
        out_name = os.path.splitext(os.path.basename(path))[0] + ".png"
        out_path = os.path.join(output_dir, out_name)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
