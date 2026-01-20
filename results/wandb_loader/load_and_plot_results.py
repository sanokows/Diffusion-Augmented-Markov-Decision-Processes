#!/usr/bin/env python3

import argparse
from collections import defaultdict
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import wandb


DEFAULT_Y_KEY = "eval/episode_return"
AUTO_X_KEYS = ["_step"]  # ["frame", "_step", "step", "global_step", "time_step", "time_steps"]
PROJECT_SUFFIXES = ["_FR_16_01", "_FR_19_01"]
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
            "dime_<env_name><suffix> for each suffix in PROJECT_SUFFIXES."
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
        "--run-name-exclude",
        nargs="*",
        default=["reppo-dmerl"],
        help="Optional substrings to exclude runs by name (space-separated).",
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
    print("printing df")
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


def clean_run_name(run_name: str) -> str:
    cleaned = re.sub(r"debug[\W_]*1", "", run_name, flags=re.IGNORECASE)
    for suffix in PROJECT_SUFFIXES:
        if suffix:
            cleaned = cleaned.replace(suffix, "")
    cleaned_lower = cleaned.lower()
    for env_candidate in ENV_NAMES:
        env_lower = env_candidate.lower()
        idx = cleaned_lower.find(env_lower)
        if idx != -1:
            cleaned = cleaned[:idx]
            break
    return cleaned.rstrip("_- ")


def build_distinct_palette(count: int) -> list[tuple[float, float, float, float]]:
    if count <= 0:
        return []
    palette = []
    for idx in range(count):
        hue = idx / count
        palette.append(plt.cm.hsv(hue))
    return palette


def main() -> int:
    args = parse_args()
    api = wandb.Api()
    if args.project is None:
        env_projects = {
            env_name: [f"dime_{env_name}{suffix}" for suffix in PROJECT_SUFFIXES]
            for env_name in ENV_NAMES
        }
        multi_env = True
    else:
        env_projects = {args.project: [args.project]}
        multi_env = False

    figures_dir = os.path.join("results", "wandb_loader", "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    filters = {"state": args.state} if args.state else {}
    overall_records = []
    env_results = []

    for env_name, projects in env_projects.items():
        records = []
        group_run_counts = defaultdict(set)
        skipped = 0

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

            suffix = project
            env_prefix = f"dime_{env_name}"
            if project.startswith(env_prefix):
                suffix = project[len(env_prefix) :] or project

            for run in runs:
                run_name = clean_run_name(run.name or run.id)
                print(run_name)
                if args.run_name_exclude:
                    if any(substr in run_name for substr in args.run_name_exclude):
                        continue
                if args.run_name_contains and args.run_name_contains not in run_name:
                    continue
                df = load_run_history(run, args.y_key, args.x_key)
                if df is None:
                    skipped += 1
                    continue
                group_name = f"{run_name} {suffix}"
                df["group"] = group_name
                df["run_name"] = run_name
                df["suffix"] = suffix
                df["run_id"] = run.id
                records.append(df)
                group_run_counts[group_name].add(run.id)

        if not records:
            print(
                f"No runs contained both {args.y_key} and a usable step key in {env_name}.",
                file=sys.stderr,
            )
            continue

        data = pd.concat(records, ignore_index=True)
        overall_records.append(data)
        env_results.append(
            {
                "env_name": env_name,
                "data": data,
                "group_run_counts": group_run_counts,
                "skipped": skipped,
            }
        )

    color_by_run_name = {}
    linestyle_by_suffix = {}
    if overall_records:
        overall_frame = pd.concat(overall_records, ignore_index=True)
        all_run_names = sorted(overall_frame["run_name"].unique())
        all_suffixes = sorted(overall_frame["suffix"].unique())
        group_to_run_name = (
            overall_frame[["group", "run_name"]]
            .drop_duplicates()
            .set_index("group")["run_name"]
            .to_dict()
        )
        group_to_suffix = (
            overall_frame[["group", "suffix"]]
            .drop_duplicates()
            .set_index("group")["suffix"]
            .to_dict()
        )
        palette = build_distinct_palette(len(all_run_names))
        for idx, run_name in enumerate(all_run_names):
            color_by_run_name[run_name] = palette[idx % len(palette)]
        line_styles = ["-", "--", "-.", ":"]
        for idx, suffix in enumerate(all_suffixes):
            linestyle_by_suffix[suffix] = line_styles[idx % len(line_styles)]

        for env_result in env_results:
            env_name = env_result["env_name"]
            data = env_result["data"]
            group_run_counts = env_result["group_run_counts"]
            skipped = env_result["skipped"]
            group_to_run_name = (
                data[["group", "run_name"]]
                .drop_duplicates()
                .set_index("group")["run_name"]
                .to_dict()
            )
            group_to_suffix = (
                data[["group", "suffix"]]
                .drop_duplicates()
                .set_index("group")["suffix"]
                .to_dict()
            )

        grouped = data.groupby(["group", "step"])["value"]
        mean = grouped.mean().reset_index()
        stderr = grouped.sem().reset_index().rename(columns={"value": "stderr"})
        merged = mean.merge(stderr, on=["group", "step"], how="left")

        plt.figure(figsize=(9, 5))
        labeled_run_names = set()
        for group, group_df in merged.groupby("group"):
            group_df = group_df.sort_values("step")
            run_count = len(group_run_counts[group])
            run_name = group_to_run_name.get(group, group)
            suffix = group_to_suffix.get(group, "")
            label = (
                f"{run_name} (n={run_count})"
                if run_name not in labeled_run_names
                else "_nolegend_"
            )
            labeled_run_names.add(run_name)
            color = color_by_run_name.get(run_name)
            linestyle = linestyle_by_suffix.get(suffix, "-")
            plt.plot(
                group_df["step"],
                group_df["value"],
                label=label,
                color=color,
                linestyle=linestyle,
            )
            if group_df["stderr"].notna().any():
                plt.fill_between(
                    group_df["step"],
                    group_df["value"] - group_df["stderr"],
                    group_df["value"] + group_df["stderr"],
                    color=color,
                    alpha=0.2,
                )

        plt.xlabel("env calls (step)")
        plt.ylabel(args.y_key)
        plt.title(f"{env_name}: {args.y_key} (grouped by run name + suffix)")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()

        if args.out:
            if multi_env:
                base, ext = os.path.splitext(args.out)
                output_path = f"{base}_{env_name}{ext}"
            else:
                output_path = args.out
        else:
            output_name = (
                f"dime_{env_name}_multi_suffix_avg_eval_return.png"
                if multi_env
                else f"{env_name}_avg_eval_return.png"
            )
            output_path = os.path.join(figures_dir, output_name)

        plt.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")

        per_run_grouped = data.groupby("step")["value"]
        per_run_mean = per_run_grouped.mean().reset_index()
        per_run_std = per_run_grouped.std().reset_index().rename(columns={"value": "std"})
        per_run_merged = per_run_mean.merge(per_run_std, on="step", how="left")

        plt.figure(figsize=(9, 5))
        for run_id, run_df in data.groupby("run_id"):
            run_df = run_df.sort_values("step")
            run_name = run_df["run_name"].iloc[0]
            suffix = run_df["suffix"].iloc[0]
            color = color_by_run_name.get(run_name)
            linestyle = linestyle_by_suffix.get(suffix, "-")
            plt.plot(
                run_df["step"],
                run_df["value"],
                alpha=0.25,
                linewidth=1,
                color=color,
                linestyle=linestyle,
            )

        per_run_merged = per_run_merged.sort_values("step")
        plt.plot(per_run_merged["step"], per_run_merged["value"], color="black", label="Mean")
        if per_run_merged["std"].notna().any():
            plt.fill_between(
                per_run_merged["step"],
                per_run_merged["value"] - per_run_merged["std"],
                per_run_merged["value"] + per_run_merged["std"],
                color="black",
                alpha=0.15,
                label="Std",
            )

        plt.xlabel("env calls (step)")
        plt.ylabel(args.y_key)
        plt.title(f"{env_name}: {args.y_key} (per-run + mean/std)")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()

        output_name = (
            f"dime_{env_name}_multi_suffix_runs_avg_std.png"
            if multi_env
            else f"{env_name}_runs_avg_std.png"
        )
        output_path = os.path.join(figures_dir, output_name)
        plt.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")

        if skipped:
            print(f"Skipped {skipped} runs without usable data in {env_name}.")

    if len(env_projects) > 1 and overall_records:
        overall_data = pd.concat(overall_records, ignore_index=True)
        overall_grouped = overall_data.groupby(["group", "step"])["value"]
        overall_mean = overall_grouped.mean().reset_index()
        overall_stderr = overall_grouped.sem().reset_index().rename(columns={"value": "stderr"})
        overall_merged = overall_mean.merge(overall_stderr, on=["group", "step"], how="left")
        overall_group_counts = overall_data.groupby("group")["run_id"].nunique().to_dict()

        plt.figure(figsize=(9, 5))
        labeled_run_names = set()
        for group, group_df in overall_merged.groupby("group"):
            group_df = group_df.sort_values("step")
            run_count = overall_group_counts.get(group, 0)
            run_name = group_to_run_name.get(group, group)
            suffix = group_to_suffix.get(group, "")
            label = (
                f"{run_name} (n={run_count})"
                if run_name not in labeled_run_names
                else "_nolegend_"
            )
            labeled_run_names.add(run_name)
            color = color_by_run_name.get(run_name)
            linestyle = linestyle_by_suffix.get(suffix, "-")
            plt.plot(
                group_df["step"],
                group_df["value"],
                label=label,
                color=color,
                linestyle=linestyle,
            )
            if group_df["stderr"].notna().any():
                plt.fill_between(
                    group_df["step"],
                    group_df["value"] - group_df["stderr"],
                    group_df["value"] + group_df["stderr"],
                    color=color,
                    alpha=0.2,
                )

        plt.xlabel("env calls (step)")
        plt.ylabel(args.y_key)
        plt.title(f"All environments: {args.y_key} (grouped by run name + suffix)")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()

        output_path = os.path.join(figures_dir, "all_envs_multi_suffix_avg_eval_return.png")
        plt.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
