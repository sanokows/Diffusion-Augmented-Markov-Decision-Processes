#!/usr/bin/env python3

import argparse
from collections import defaultdict
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import wandb


DEFAULT_Y_KEY = "eval/episode_return"
AUTO_X_KEYS = ["_step"]
ENV_NAMES = [
    "AcrobotSwingup",
    "BallInCup",
    #"AcrobotSwingupSparse",
    #"PendulumSwingup",
]

PARAM_SPECS = [
    ("lr", ["hyperparameters.lr", "lr"]),
    ("gamma", ["hyperparameters.gamma", "gamma"]),
    ("lmbda", ["hyperparameters.lmbda", "lmbda"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load W&B runs from hyperparameter sweeps, group by swept "
            "hyperparameters, and plot average eval/episode_return."
        )
    )
    parser.add_argument(
        "--project",
        default=None,
        help=(
            "Optional W&B project name. If omitted, loops over ENV_NAMES using "
            "dime_<env_name>_HYPERPARAMS."
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
        "--out-dir",
        default=os.path.join("results", "wandb_loader", "Figures", "hyperparameters"),
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--run-name-contains",
        default=None,
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
    if df.empty:
        return None
    x_resolved = resolve_x_key(df, x_key)
    if x_resolved is None or y_key not in df.columns:
        return None
    df = df[[x_resolved, y_key]].dropna()
    if df.empty:
        return None
    df = df.rename(columns={x_resolved: "step", y_key: "value"})
    return df


def get_config_value(config: dict, key: str):
    if key in config:
        return config[key]
    if "." not in key:
        return None
    current = config
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def format_value(value) -> str:
    value_float = to_float(value)
    if value_float is not None:
        return f"{value_float:g}"
    return str(value)


def build_group_label(config: dict) -> str:
    labels = []
    for name, keys in PARAM_SPECS:
        value = None
        for key in keys:
            value = get_config_value(config, key)
            if value is not None:
                break
        if value is None:
            continue
        labels.append(f"{name}={format_value(value)}")
    if labels:
        return ", ".join(labels)
    return "unknown"


def infer_env_name(config: dict, project_name: str) -> str | None:
    for key in ["env.name", "env_name", "env"]:
        value = get_config_value(config, key)
        if value:
            return str(value)
    project_base = project_name.split("/")[-1]
    if project_base.startswith("dime_") and project_base.endswith("_HYPERPARAMS"):
        return project_base[len("dime_") : -len("_HYPERPARAMS")]
    return None


def plot_grouped_data(
    grouped_df: pd.DataFrame,
    group_counts: dict,
    title: str,
    y_key: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(9, 5))
    for group, group_df in grouped_df.groupby("group"):
        group_df = group_df.sort_values("step")
        count = group_counts.get(group, 0)
        label = f"{group} (n={count})"
        plt.plot(group_df["step"], group_df["value"], label=label)
        if group_df["stderr"].notna().any():
            plt.fill_between(
                group_df["step"],
                group_df["value"] - group_df["stderr"],
                group_df["value"] + group_df["stderr"],
                alpha=0.2,
            )
    plt.xlabel("env calls (step)")
    plt.ylabel(y_key)
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


def main() -> int:
    args = parse_args()
    api = wandb.Api()
    if args.project is None:
        projects = [f"dime_{env_name}_HYPERPARAMS" for env_name in ENV_NAMES]
    else:
        projects = [args.project]

    os.makedirs(args.out_dir, exist_ok=True)
    filters = {"state": args.state} if args.state else {}
    safe_y_key = args.y_key.replace("/", "_")

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

        for run in runs:
            run_name = run.name or ""
            if args.run_name_contains and args.run_name_contains not in run_name:
                continue
            df = load_run_history(run, args.y_key, args.x_key)
            if df is None:
                skipped += 1
                continue

            config = run.config or {}
            env_name = infer_env_name(config, project)
            if env_name is None:
                skipped += 1
                continue

            group_label = build_group_label(config)
            df["group"] = group_label
            df["env"] = env_name
            df["run_id"] = run.id
            records.append(df)
            group_run_counts[(env_name, group_label)].add(run.id)

    if not records:
        print("No runs contained usable data.", file=sys.stderr)
        return 1

    data = pd.concat(records, ignore_index=True)

    for env_name in sorted(data["env"].unique()):
        env_data = data[data["env"] == env_name]
        grouped = env_data.groupby(["group", "step"])["value"]
        mean = grouped.mean().reset_index()
        stderr = grouped.sem().reset_index().rename(columns={"value": "stderr"})
        merged = mean.merge(stderr, on=["group", "step"], how="left")
        group_counts = {
            group: len(group_run_counts[(env_name, group)])
            for group in merged["group"].unique()
        }
        output_path = os.path.join(
            args.out_dir, f"{env_name}_hyperparams_{safe_y_key}.png"
        )
        title = f"{env_name}: {args.y_key} (grouped by hyperparameters)"
        plot_grouped_data(merged, group_counts, title, args.y_key, output_path)

    env_mean = data.groupby(["env", "group", "step"])["value"].mean().reset_index()
    overall_grouped = env_mean.groupby(["group", "step"])["value"]
    overall_mean = overall_grouped.mean().reset_index()
    overall_stderr = overall_grouped.sem().reset_index().rename(columns={"value": "stderr"})
    overall_merged = overall_mean.merge(overall_stderr, on=["group", "step"], how="left")
    group_env_counts = env_mean.groupby("group")["env"].nunique().to_dict()
    overall_output = os.path.join(args.out_dir, f"overall_hyperparams_{safe_y_key}.png")
    plot_grouped_data(
        overall_merged,
        group_env_counts,
        f"Overall: {args.y_key} (avg over envs)",
        args.y_key,
        overall_output,
    )

    if skipped:
        print(f"Skipped {skipped} runs without usable data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
