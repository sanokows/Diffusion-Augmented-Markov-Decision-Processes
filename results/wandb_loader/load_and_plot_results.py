#!/usr/bin/env python3

import argparse
from collections import defaultdict
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import wandb


DEFAULT_Y_KEY = "eval/episode_return"
AUTO_X_KEYS = ["_step"]  # ["frame", "_step", "step", "global_step", "time_step", "time_steps"]
ENV_NAMES = [
    "AcrobotSwingup",
    "PendulumSwingup",
    "CartpoleSwingupSparse",
    "AcrobotSwingupSparse",
     "CheetahRun",
    "FishSwim",
    "HopperHop",
    "HopperStand",
    "WalkerRun",
    "WalkerStand",
    "WalkerWalk",
    "FingerSpin"   
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load wandb runs from a project, group by run name, and plot the "
            "average eval/episode_return over env-call steps."
        )
    )
    parser.add_argument(
        "--project",
        default=None,
        help=(
            "Optional W&B project name. If omitted, loops over ENV_NAMES using "
            "dime_<env_name>_FR_16_01."
        ),
    )
    parser.add_argument("--entity", default="sanokows", help="W&B entity (team/user).")
    parser.add_argument(
        "--y-key",
        default=DEFAULT_Y_KEY,
        help="Metric to plot on Y axis.",
    )
    parser.add_argument(
        "--x-key",
        default="auto",
        help=(
            "Metric to use for X axis. Use 'auto' to try common step keys "
            f"({', '.join(AUTO_X_KEYS)})."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for the plot (png/pdf). If not set, shows the plot.",
    )
    parser.add_argument(
        "--run-name-contains",
        default=None,
        help="Optional substring to filter runs by name.",
    )
    parser.add_argument(
        "--run-name",
        default="reppo-dime",
        help="Optional substring to filter runs by name.",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Optional run state filter (e.g., finished, running).",
    )
    return parser.parse_args()


def resolve_project_path(api: wandb.Api, entity: str | None, project: str) -> str:
    if entity:
        return f"{entity}/{project}"
    default_entity = getattr(api, "default_entity", None)
    if default_entity:
        return f"{default_entity}/{project}"
    return project


def resolve_x_key(df: pd.DataFrame, x_key: str) -> str | None:
    if x_key != "auto":
        return x_key if x_key in df.columns else None
    for key in AUTO_X_KEYS:
        if key in df.columns:
            return key
    return None


def load_run_history(run: wandb.apis.public.Run, y_key: str, x_key: str) -> pd.DataFrame | None:
    keys = [y_key]
    if x_key != "auto":
        keys.append(x_key)
    try:
        df = run.history(keys=keys, pandas=True)
    except ValueError as exc:
        print(
            f"Warning: could not load history for run {run.name or run.id} ({exc}).",
            file=sys.stderr,
        )
        return None
    print(df)
    print(df.columns)
    if df.empty:
        return None
    x_resolved = resolve_x_key(df, x_key)
    if x_resolved is None:
        return None
    if y_key not in df.columns:
        return None
    df = df[[x_resolved, y_key]].dropna()
    if df.empty:
        return None
    df = df.rename(columns={x_resolved: "step", y_key: "value"})
    return df


def main() -> int:
    args = parse_args()
    api = wandb.Api()
    if args.project is None:
        projects = [f"dime_{env_name}_FR_16_01" for env_name in ENV_NAMES]
    else:
        projects = [args.project]

    figures_dir = os.path.join("results", "wandb_loader", "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    filters = {"state": args.state} if args.state else {}
    overall_records = []

    for project in projects:
        project_path = resolve_project_path(api, args.entity, project)

        try:
            runs = api.runs(project_path, filters=filters or None)
        except Exception as exc:
            print(
                f"Warning: project {project_path} not found or inaccessible ({exc}).",
                file=sys.stderr,
            )
            continue
        if not runs:
            print(f"Warning: no runs found for project {project_path}.", file=sys.stderr)
            continue

        records = []
        group_run_counts = defaultdict(set)
        skipped = 0

        for run in runs:
            run_name = run.name or run.id
            print(run_name)
            if args.run_name and args.run_name not in run_name:
                continue
            if args.run_name_contains and args.run_name_contains not in run_name:
                continue

            df = load_run_history(run, args.y_key, args.x_key)
            print(df)
            if df is None:
                skipped += 1
                continue
            df["group"] = run.name or run.id
            df["run_id"] = run.id
            records.append(df)
            group_run_counts[run.name or run.id].add(run.id)

        if not records:
            print(
                f"No runs contained both {args.y_key} and a usable step key in {project}.",
                file=sys.stderr,
            )
            continue

        data = pd.concat(records, ignore_index=True)
        overall_records.append(data)
        grouped = data.groupby(["group", "step"])["value"]
        mean = grouped.mean().reset_index()
        stderr = grouped.sem().reset_index().rename(columns={"value": "stderr"})
        merged = mean.merge(stderr, on=["group", "step"], how="left")

        plt.figure(figsize=(9, 5))
        for group, group_df in merged.groupby("group"):
            group_df = group_df.sort_values("step")
            run_count = len(group_run_counts[group])
            label = f"{group} (n={run_count})"
            plt.plot(group_df["step"], group_df["value"], label=label)
            if group_df["stderr"].notna().any():
                plt.fill_between(
                    group_df["step"],
                    group_df["value"] - group_df["stderr"],
                    group_df["value"] + group_df["stderr"],
                    alpha=0.2,
                )

        plt.xlabel("env calls (step)")
        plt.ylabel(args.y_key)
        plt.title(f"{project}: {args.y_key} (grouped by run name)")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()

        if args.out:
            output_path = args.out
        else:
            output_path = os.path.join(figures_dir, f"{project}_avg_eval_return.png")

        plt.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")

        if skipped:
            print(f"Skipped {skipped} runs without usable data in {project}.")

    if len(projects) > 1 and overall_records:
        overall_data = pd.concat(overall_records, ignore_index=True)
        overall_grouped = overall_data.groupby("step")["value"]
        overall_mean = overall_grouped.mean().reset_index()
        overall_stderr = overall_grouped.sem().reset_index().rename(columns={"value": "stderr"})
        overall_merged = overall_mean.merge(overall_stderr, on="step", how="left")

        plt.figure(figsize=(9, 5))
        overall_merged = overall_merged.sort_values("step")
        plt.plot(overall_merged["step"], overall_merged["value"], label="All environments")
        if overall_merged["stderr"].notna().any():
            plt.fill_between(
                overall_merged["step"],
                overall_merged["value"] - overall_merged["stderr"],
                overall_merged["value"] + overall_merged["stderr"],
                alpha=0.2,
            )

        plt.xlabel("env calls (step)")
        plt.ylabel(args.y_key)
        plt.title(f"All environments: {args.y_key}")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()

        output_path = os.path.join(figures_dir, "all_envs_avg_eval_return.png")
        plt.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
