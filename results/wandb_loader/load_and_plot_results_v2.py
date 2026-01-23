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
AUTO_X_KEYS = ["_step"]
PROJECT_SUFFIXES = ["_FR_16_01", "_FR_19_01"]
ENV_NAMES = [
        "WalkerStand",
    "AcrobotSwingup",
    "PendulumSwingup",
    "CartpoleSwingupSparse",
    "AcrobotSwingupSparse",
    "CheetahRun",
    "FishSwim",
    "HopperHop",
    "HopperStand",
    "WalkerRun",
    "WalkerWalk",
    "FingerSpin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load W&B runs, group by method (from run name), and plot "
            "eval/episode_return over steps for each environment."
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
    parser.add_argument("--y-key", default=DEFAULT_Y_KEY, help="Metric to plot on Y axis.")
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
        help="Output path for plots (png/pdf). Env name is appended when multiple envs.",
    )
    parser.add_argument(
        "--run-name-contains",
        default=None,
        help="Optional substring to filter runs by name.",
    )
    parser.add_argument(
        "--run-name-exclude",
        nargs="*",
        default=["reppo_dmerl"],
        help="Optional substrings to exclude runs by name (space-separated).",
    )
    parser.add_argument(
        "--state",
        default="finished",
        help="Optional run state filter (e.g., finished, running).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print available history keys when a run is skipped.",
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


def clean_method_name(raw_name: str) -> str:
    cleaned = re.sub(r"debug[\W_]*1", "", raw_name, flags=re.IGNORECASE)
    for suffix in PROJECT_SUFFIXES:
        if suffix:
            cleaned = cleaned.replace(suffix, "")
    cleaned_lower = cleaned.lower()
    for env_name in ENV_NAMES:
        env_lower = env_name.lower()
        idx = cleaned_lower.find(env_lower)
        if idx != -1:
            cleaned = cleaned[:idx]
            cleaned_lower = cleaned.lower()
            break
    cleaned = re.sub(r"[_\- ]+", "_", cleaned).strip("_-")
    return cleaned or raw_name


def build_distinct_palette(count: int) -> list[tuple[float, float, float, float]]:
    if count <= 0:
        return []
    palette = []
    for idx in range(count):
        hue = idx / count
        palette.append(plt.cm.hsv(hue))
    return palette


def load_run_history(
    run: wandb.apis.public.Run,
    y_key: str,
    x_key: str,
    verbose: bool,
) -> pd.DataFrame | None:
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
    if df.empty:
        return None
    x_resolved = resolve_x_key(df, x_key)
    if x_resolved is None or y_key not in df.columns:
        if verbose:
            available = ", ".join(sorted(df.columns))
            print(
                f"Warning: missing keys for run {run.name or run.id}. "
                f"Available keys: {available}",
                file=sys.stderr,
            )
        return None
    df = df[[x_resolved, y_key]].dropna()
    if df.empty:
        return None
    df = df.rename(columns={x_resolved: "step", y_key: "value"})
    return df


def pad_hopperstand_zero_tail(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    values = df["value"].to_numpy()
    if values.size == 0:
        return df, False
    has_good = False
    tail_start = None
    for idx, val in enumerate(values):
        if val > 1:
            has_good = True
        if has_good and val == 0 and (values[idx:] == 0).all():
            tail_start = idx
            break
    if tail_start is None or tail_start == 0:
        return df, False
    df = df.copy()
    df.loc[df.index[tail_start:], "value"] = values[tail_start - 1]
    return df, True


def infer_env_from_project(project: str) -> str:
    for env_name in ENV_NAMES:
        if env_name.lower() in project.lower():
            return env_name
    return project


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
        env_name = infer_env_from_project(args.project)
        env_projects = {env_name: [args.project]}
        multi_env = False

    figures_dir = os.path.join("results", "wandb_loader", "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    filters = {"state": args.state} if args.state else {}

    env_results = []
    all_methods = set()

    for env_name, projects in env_projects.items():
        records = []
        method_run_counts = defaultdict(set)
        skipped = 0

        for project in projects:
            project_path = resolve_project_path(api, args.entity, project)
            run_name_counts = defaultdict(int)
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

            for run in runs:
                raw_name = run.name or run.id
                if project.endswith("_FR_19_01") and raw_name.strip().endswith("WPO"):
                    continue
                if args.run_name_exclude and any(substr in raw_name for substr in args.run_name_exclude):
                    continue
                if args.run_name_contains and args.run_name_contains not in raw_name:
                    continue
                run_name_counts[raw_name] += 1

                df = load_run_history(run, args.y_key, args.x_key, args.verbose)
                if df is None:
                    skipped += 1
                    continue
                print(f" {env_name} - {raw_name}: {df.shape}")
                if env_name == "HopperStand":
                    ### print the shape of df before and after padding
                    print(f"Before padding: {df.shape}")
                    df, padded = pad_hopperstand_zero_tail(df)
                    print(f"After padding: {df.shape}")
                    if padded:
                        print(
                            f"Warning: padded zero tail in HopperStand for run "
                            f"{run.name or run.id}.",
                            file=sys.stderr,
                        )

                method_name = clean_method_name(raw_name)
                df["method"] = method_name
                df["run_id"] = run.id
                records.append(df)
                method_run_counts[method_name].add(run.id)
                all_methods.add(method_name)

            duplicate_counts = {
                name: count for name, count in run_name_counts.items() if count > 1
            }
            if duplicate_counts:
                print(f"Duplicate run names in {project_path}:")
                for name, count in sorted(duplicate_counts.items()):
                    print(f"  {name}: {count}")
            else:
                print(f"No duplicate run names in {project_path}.")

        if not records:
            print(
                f"No runs contained both {args.y_key} and a usable step key in {env_name}.",
                file=sys.stderr,
            )
            continue

        env_results.append(
            {
                "env_name": env_name,
                "records": pd.concat(records, ignore_index=True),
                "method_run_counts": method_run_counts,
                "skipped": skipped,
            }
        )

    color_by_method = {}
    palette = build_distinct_palette(len(all_methods))
    for idx, method in enumerate(sorted(all_methods)):
        color_by_method[method] = palette[idx % len(palette)]

    for result in env_results:
        env_name = result["env_name"]
        data = result["records"]
        method_run_counts = result["method_run_counts"]
        skipped = result["skipped"]

        grouped = data.groupby(["method", "step"])["value"]
        mean = grouped.mean().reset_index()
        stderr = grouped.sem().reset_index().rename(columns={"value": "stderr"})
        merged = mean.merge(stderr, on=["method", "step"], how="left")

        plt.figure(figsize=(9, 5))
        for method, method_df in merged.groupby("method"):
            method_df = method_df.sort_values("step")
            run_count = len(method_run_counts[method])
            label = f"{method} (n={run_count})"
            color = color_by_method.get(method)
            plt.plot(method_df["step"], method_df["value"], label=label, color=color)
            if method_df["stderr"].notna().any():
                plt.fill_between(
                    method_df["step"],
                    method_df["value"] - method_df["stderr"],
                    method_df["value"] + method_df["stderr"],
                    color=color,
                    alpha=0.2,
                )

        plt.xlabel("env calls (step)")
        plt.ylabel(args.y_key)
        plt.title(f"{env_name}: {args.y_key} (grouped by method)")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()

        if args.out:
            if multi_env:
                base, ext = os.path.splitext(args.out)
                output_path = f"{base}_{env_name}{ext}"
            else:
                output_path = args.out
        else:
            output_path = os.path.join(
                figures_dir, f"dime_{env_name}_methods_avg_eval_return.png"
            )

        plt.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")

        if skipped:
            print(f"Skipped {skipped} runs without usable data in {env_name}.")

    if multi_env and env_results:
        overall_records = pd.concat([r["records"] for r in env_results], ignore_index=True)
        overall_grouped = overall_records.groupby(["method", "step"])["value"]
        overall_mean = overall_grouped.mean().reset_index()
        overall_stderr = overall_grouped.sem().reset_index().rename(columns={"value": "stderr"})
        overall_merged = overall_mean.merge(overall_stderr, on=["method", "step"], how="left")

        overall_run_counts = overall_records.groupby("method")["run_id"].nunique().to_dict()

        plt.figure(figsize=(9, 5))
        for method, method_df in overall_merged.groupby("method"):
            method_df = method_df.sort_values("step")
            run_count = overall_run_counts.get(method, 0)
            label = f"{method} (n={run_count})"
            color = color_by_method.get(method)
            plt.plot(method_df["step"], method_df["value"], label=label, color=color)
            if method_df["stderr"].notna().any():
                plt.fill_between(
                    method_df["step"],
                    method_df["value"] - method_df["stderr"],
                    method_df["value"] + method_df["stderr"],
                    color=color,
                    alpha=0.2,
                )

        plt.xlabel("env calls (step)")
        plt.ylabel(args.y_key)
        plt.title(f"All environments: {args.y_key} (grouped by method)")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()

        output_path = os.path.join(figures_dir, "all_envs_methods_avg_eval_return.png")
        plt.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
