#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from timing_wandb_utils import (
    METHOD_ORDER,
    aggregate_env_method,
    collect_timing_data,
    resolve_projects,
    split_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load W&B timing logs, discard warmup points, and plot timing stats per env/method."
        )
    )
    parser.add_argument("--entity", default="", help="W&B entity/team. Empty uses default.")
    parser.add_argument(
        "--projects",
        default="",
        help="Comma-separated explicit project names. If empty, uses project_prefix + env_names.",
    )
    parser.add_argument(
        "--project-prefix",
        default="dime_",
        help="Project prefix when --projects is not provided.",
    )
    parser.add_argument(
        "--env-names",
        required=True,
        help="Comma-separated env names used to resolve projects when --projects is empty.",
    )
    parser.add_argument(
        "--methods",
        default="reppo,ppo,dime,diffppo,dmerl",
        help="Comma-separated canonical methods to keep.",
    )
    parser.add_argument(
        "--state",
        default="finished",
        help="W&B run state filter (finished, running, any).",
    )
    parser.add_argument(
        "--drop-first",
        type=int,
        default=1,
        help="Number of initial timing points to discard per run (JIT warmup).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=1,
        help="Minimum remaining timing points required to keep a run.",
    )
    parser.add_argument(
        "--out-dir",
        default="READMEs/Timing/outputs",
        help="Output directory for CSV and plot artifacts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skipped runs and fetch issues.",
    )
    return parser.parse_args()


def _plot_env_bars(env_df, out_path: Path, env_name: str) -> None:
    if env_df.empty:
        return

    env_df = env_df.copy()
    env_df["method_order"] = env_df["method"].map(
        {name: idx for idx, name in enumerate(METHOD_ORDER)}
    ).fillna(1e9)
    env_df = env_df.sort_values(["method_order", "method"]).reset_index(drop=True)

    labels = env_df["method_display"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    rollout = env_df["rollout_mean"].to_numpy()
    update = env_df["update_mean"].to_numpy()
    total = env_df["total_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 4.8))
    ax.bar(x - width, rollout, width, label="rollout")
    ax.bar(x, update, width, label="update")
    ax.bar(x + width, total, width, label="total")

    ax.set_title(f"Timing Means per Method - {env_name}")
    ax.set_ylabel("seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_env_integrated_bars(env_df, out_path: Path, env_name: str) -> None:
    if env_df.empty:
        return

    env_df = env_df.copy()
    env_df["method_order"] = env_df["method"].map(
        {name: idx for idx, name in enumerate(METHOD_ORDER)}
    ).fillna(1e9)
    env_df = env_df.sort_values(["method_order", "method"]).reset_index(drop=True)

    labels = env_df["method_display"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    rollout = env_df["rollout_integrated_mean"].to_numpy()
    update = env_df["update_integrated_mean"].to_numpy()
    total = env_df["total_integrated_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 4.8))
    ax.bar(x - width, rollout, width, label="rollout integrated")
    ax.bar(x, update, width, label="update integrated")
    ax.bar(x + width, total, width, label="total integrated")

    ax.set_title(f"Integrated Timing per Method - {env_name}")
    ax.set_ylabel("seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_env_sps_bars(env_df, out_path: Path, env_name: str) -> None:
    if env_df.empty:
        return

    env_df = env_df.copy()
    env_df["method_order"] = env_df["method"].map(
        {name: idx for idx, name in enumerate(METHOD_ORDER)}
    ).fillna(1e9)
    env_df = env_df.sort_values(["method_order", "method"]).reset_index(drop=True)

    labels = env_df["method_display"].tolist()
    x = np.arange(len(labels))
    sps = env_df["sps_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    ax.bar(x, sps, width=0.6, label="sps")
    ax.set_title(f"SPS per Method - {env_name}")
    ax.set_ylabel("steps / second")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    projects = split_csv(args.projects)
    env_names = split_csv(args.env_names)
    methods = set(split_csv(args.methods))
    resolved_projects = resolve_projects(projects, env_names, args.project_prefix)

    collected = collect_timing_data(
        entity=args.entity,
        projects=resolved_projects,
        allowed_methods=methods,
        state=args.state,
        drop_first=args.drop_first,
        min_points=args.min_points,
        verbose=args.verbose,
    )
    agg_df = aggregate_env_method(collected.run_summary_df)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_csv = out_dir / "timing_run_summary.csv"
    agg_csv = out_dir / "timing_env_method_summary.csv"
    collected.run_summary_df.to_csv(run_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)

    if not agg_df.empty and "env_name" in agg_df.columns:
        for env_name, env_df in agg_df.groupby("env_name"):
            out_png_mean = out_dir / f"timing_means_{env_name}.png"
            out_png_integrated = out_dir / f"timing_integrated_{env_name}.png"
            out_png_sps = out_dir / f"timing_sps_{env_name}.png"
            _plot_env_bars(env_df, out_png_mean, env_name)
            _plot_env_integrated_bars(env_df, out_png_integrated, env_name)
            _plot_env_sps_bars(env_df, out_png_sps, env_name)

    print(f"[summary] projects={resolved_projects}")
    print(f"[summary] run summary rows: {len(collected.run_summary_df)} -> {run_csv}")
    print(f"[summary] env-method rows: {len(agg_df)} -> {agg_csv}")
    print(f"[summary] per-env plots saved under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
