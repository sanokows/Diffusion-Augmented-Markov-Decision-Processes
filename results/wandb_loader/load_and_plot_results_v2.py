#!/usr/bin/env python3

import argparse
from collections import defaultdict
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb


DEFAULT_Y_KEY = "eval/episode_return"
AUTO_X_KEYS = ["_step"]
PROJECT_SUFFIXES = ["_FR_16_01", "_FR_19_01", "_FR_24_01", "_FR_30_01", "_FR_test_PPO", "_FR_ME-WPO", "_FR_WPO"]
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
# Optional per-suffix run selection with aliases.
# Format: { "<suffix>": { "<run_name>": "<alias>", ... }, ... }
# Leave empty or omit suffix keys to use default name-based grouping.
#RUNS_BY_SUFFIX: dict[str, dict[str, str]] = {}
RUNS_BY_SUFFIX = {
    "_FR_16_01": {
        "reppo-dime-debug-1-<env_name>": "REPPO-DiME" ,
    },
    "_FR_19_01": {
        "reppo-<env_name>-reparam": "REPPO",
    },
    "_FR_24_01": {
        "reppo-dmerl-debug-1-<env_name>-WPO": "DME-WPO (ours)",
    },
    "_FR_30_01": {
        "reppo-dmerl-debug-1-<env_name>-reparam": "DME-REPPO (ours)",
    },
    "_FR_test_PPO": {
        "ppo-diff_ppo-<env_name>": "DME-PPO (ours)",
    },
    "_FR_ME-WPO": {
        "reppo-<env_name>-WPO": "ME-WPO (ours)",
    },
    "_FR_WPO": {
        "reppo-<env_name>-WPO": "WPO",
    },
}
PPO_BRAX_LABEL = "PPO (r)"
CSV_RESULTS_DIR = Path("results")
PLOT_MAX_STEPS = 5e7
COLORBLIND_PALETTE = [
  #  "#0072B2",  # blue
    "#FF000D",  # orange #red
    "#FFAE00",  # orange
    "#00FF00",  # green
    "#FC04DB",  # vermillion
    "#2600FF",  # sky blue
   # "#F0E442",  # yellow
   # "#000000",  # black
  #  "#7F7F7F",  # gray
    "#835603",  # brown
    "#4C8D02",  # olive
    "#00FFFF",  # purple
    "#00885F",  # teal
]
LINE_STYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 2))]
LINE_WIDTH = 2.0
EXTRAPOLATE_ENV_NAMES = {"WalkerWalk", "WalkerRun", "WalkerStand"}
EXTRAPOLATE_MIN_FRACTION = 0.8
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 12
GRID_ALPHA = 0.7
alpha = 0.07
alpha_line = 0.8
GRID_LINESTYLE = ":"


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


def resolve_project_suffix(project: str) -> str | None:
    for suffix in PROJECT_SUFFIXES:
        if project.endswith(suffix):
            return suffix
    return None


def normalize_runs_by_suffix(
    mapping: dict[str, dict[str, str]] | None,
) -> dict[str, dict[str, str]]:
    if not mapping:
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for suffix, run_map in mapping.items():
        if not suffix or not run_map:
            continue
        clean_suffix = suffix.strip()
        if not clean_suffix:
            continue
        normalized[clean_suffix] = {
            run_name.strip(): (alias.strip() or run_name.strip())
            for run_name, alias in run_map.items()
            if run_name and run_name.strip()
        }
    return normalized


def expand_name_map(name_map: dict[str, str], env_name: str) -> dict[str, str]:
    expanded: dict[str, str] = {}
    env_lower = env_name.lower()
    for raw_name, alias in name_map.items():
        if "<env_name>" in raw_name:
            expanded[raw_name.replace("<env_name>", env_lower)] = alias
        else:
            expanded[raw_name] = alias
    return expanded


def match_run_name(
    raw_name: str, name_map: dict[str, str]
) -> tuple[str | None, str | None]:
    if raw_name in name_map:
        return name_map[raw_name], raw_name
    candidates = [key for key in name_map.keys() if key in raw_name]
    if not candidates:
        return None, None
    best_key = max(candidates, key=len)
    return name_map[best_key], best_key


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


def build_distinct_palette(count: int) -> list[str]:
    if count <= 0:
        return []
    palette = []
    for idx in range(count):
        palette.append(COLORBLIND_PALETTE[idx % len(COLORBLIND_PALETTE)])
    return palette


def _csv_paths_for_env(env_name: str) -> list[Path]:
    return sorted(CSV_RESULTS_DIR.glob(f"{env_name}*.csv"))


def _csv_tokens_for_path(path: Path, env_name: str) -> list[str]:
    stem = path.stem
    remainder = stem[len(env_name) :]
    tokens = [t for t in remainder.lower().split("_") if t]
    return tokens


def _is_csv_ppobrax_exact(path: Path, env_name: str) -> bool:
    label = _label_from_csv(path, env_name)
    return label.lower() == "ppo_brax"


def _load_csv_trials(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read CSV {path} ({exc}).", file=sys.stderr)
        return None
    if "steps" not in df.columns:
        print(f"Warning: CSV missing 'steps' column: {path}", file=sys.stderr)
        return None
    trial_cols = [c for c in df.columns if c.startswith("trial_")]
    if not trial_cols:
        print(f"Warning: CSV has no trial_* columns: {path}", file=sys.stderr)
        return None
    df = df[["steps"] + trial_cols].dropna()
    if df.empty:
        return None
    return df


def _label_from_csv(path: Path, env_name: str) -> str:
    stem = path.stem
    if stem.startswith(env_name):
        remainder = stem[len(env_name) :].lstrip("_- ")
        return remainder or "ppo"
    return stem


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


def extrapolate_last_value(
    df: pd.DataFrame, target_step: float
) -> tuple[pd.DataFrame, bool]:
    if df.empty:
        return df, False
    df = df.sort_values("step").copy()
    last_step = float(df["step"].iloc[-1])
    if last_step >= target_step:
        return df, False
    last_value = df["value"].iloc[-1]
    extrapolated = pd.DataFrame({"step": [target_step], "value": [last_value]})
    df = pd.concat([df, extrapolated], ignore_index=True)
    return df, True


def infer_env_from_project(project: str) -> str:
    for env_name in ENV_NAMES:
        if env_name.lower() in project.lower():
            return env_name
    return project


def main() -> int:
    args = parse_args()
    api = wandb.Api()
    runs_by_suffix = normalize_runs_by_suffix(RUNS_BY_SUFFIX)

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
            project_suffix = resolve_project_suffix(project)
            suffix_run_map = runs_by_suffix.get(project_suffix) if project_suffix else None
            expanded_suffix_map = (
                expand_name_map(suffix_run_map, env_name) if suffix_run_map else None
            )
            found_suffix_runs: set[str] = set()
            if expanded_suffix_map is not None:
                wanted_names = ", ".join(sorted(expanded_suffix_map.keys()))
                print(
                    f"{env_name} - {project_path}: wanted run names: {wanted_names}"
                )
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
                if expanded_suffix_map is not None:
                    print(f"{env_name} - seen run name: {raw_name}")
                if project.endswith("_FR_19_01") and raw_name.strip().endswith("WPO"):
                    continue
                if expanded_suffix_map is None:
                    if args.run_name_exclude and any(
                        substr in raw_name for substr in args.run_name_exclude
                    ):
                        continue
                    if args.run_name_contains and args.run_name_contains not in raw_name:
                        continue
                else:
                    alias, matched_key = match_run_name(raw_name, expanded_suffix_map)
                    if alias is None:
                        continue
                    print(
                        f"{env_name} - loaded run: {raw_name} "
                        f"(alias: {alias}, matched: {matched_key})"
                    )
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

                if env_name in EXTRAPOLATE_ENV_NAMES:
                    max_step = float(df["step"].max())
                    if max_step >= PLOT_MAX_STEPS * EXTRAPOLATE_MIN_FRACTION:
                        df, extrapolated = extrapolate_last_value(df, PLOT_MAX_STEPS)
                        if extrapolated:
                            print(
                                f"Warning: extrapolated {env_name} run "
                                f"{run.name or run.id} from step {max_step:.0f} "
                                f"to {PLOT_MAX_STEPS:.0f} using last value.",
                                file=sys.stderr,
                            )

                if expanded_suffix_map is None:
                    method_name = clean_method_name(raw_name)
                else:
                    method_name = alias
                    found_suffix_runs.add(matched_key or raw_name)
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

            if expanded_suffix_map is not None:
                missing = set(expanded_suffix_map.keys()) - found_suffix_runs
                if missing:
                    missing_list = ", ".join(sorted(missing))
                    print(
                        f"Warning: missing specified runs in {project_path}: {missing_list}",
                        file=sys.stderr,
                    )

        csv_paths = [
            p for p in _csv_paths_for_env(env_name) if _is_csv_ppobrax_exact(p, env_name)
        ]
        for path in csv_paths:
            df = _load_csv_trials(path)
            if df is None:
                continue
            trial_cols = [c for c in df.columns if c.startswith("trial_")]
            base_id = f"csv:{path.stem}"
            for seed_idx, col in enumerate(trial_cols):
                csv_df = pd.DataFrame({"step": df["steps"], "value": df[col]})
                csv_df["method"] = PPO_BRAX_LABEL
                csv_df["run_id"] = f"{base_id}:seed{seed_idx}"
                records.append(csv_df)
                method_run_counts[PPO_BRAX_LABEL].add(csv_df["run_id"].iloc[0])
            all_methods.add(PPO_BRAX_LABEL)

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
    style_by_method = {}
    palette = build_distinct_palette(len(all_methods))
    for idx, method in enumerate(sorted(all_methods)):
        color_by_method[method] = palette[idx % len(palette)]
        style_by_method[method] = LINE_STYLES[idx % len(LINE_STYLES)]

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
            linestyle = style_by_method.get(method, "-")
            plt.plot(
                method_df["step"],
                method_df["value"],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=LINE_WIDTH, 
                alpha = alpha_line,
            )
            if method_df["stderr"].notna().any():
                plt.fill_between(
                    method_df["step"],
                    method_df["value"] - method_df["stderr"],
                    method_df["value"] + method_df["stderr"],
                    color=color,
                    alpha=alpha,
                )

        plt.xlabel("env calls (step)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel(args.y_key, fontsize=AXIS_LABEL_FONTSIZE)
        plt.title(f"{env_name}")
        plt.legend(loc="best", fontsize=LEGEND_FONTSIZE)
        plt.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        plt.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
        plt.xlim(left=0, right=PLOT_MAX_STEPS)
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

        plt.savefig(output_path, dpi=800)
        print(f"Saved plot to {output_path}")

        if skipped:
            print(f"Skipped {skipped} runs without usable data in {env_name}.")

    if multi_env and env_results:
        overall_records = pd.concat([r["records"] for r in env_results], ignore_index=True)
        overall_grouped = overall_records.groupby(["method", "step"])["value"]
        overall_mean = overall_grouped.mean().reset_index()
        overall_stderr = overall_grouped.sem().reset_index().rename(columns={"value": "stderr"})
        overall_merged = overall_mean.merge(overall_stderr, on=["method", "step"], how="left")

        overall_method_runs: dict[str, set[str]] = defaultdict(set)
        for result in env_results:
            for method, runs in result["method_run_counts"].items():
                overall_method_runs[method].update(runs)
        overall_run_counts = {method: len(runs) for method, runs in overall_method_runs.items()}

        plt.figure(figsize=(9, 5))
        for method, method_df in overall_merged.groupby("method"):
            method_df = method_df.sort_values("step")
            run_count = overall_run_counts.get(method, 0)
            label = f"{method} (n={run_count})"
            color = color_by_method.get(method)
            linestyle = style_by_method.get(method, "-")
            plt.plot(
                method_df["step"],
                method_df["value"],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=LINE_WIDTH,
                 alpha = alpha_line,
            )
            if method_df["stderr"].notna().any():
                plt.fill_between(
                    method_df["step"],
                    method_df["value"] - method_df["stderr"],
                    method_df["value"] + method_df["stderr"],
                    color=color,
                    alpha=alpha,
                )

        plt.xlabel("env calls (step)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel(args.y_key, fontsize=AXIS_LABEL_FONTSIZE)
        plt.title(f"All environments")
        plt.legend(loc="best", fontsize=LEGEND_FONTSIZE)
        plt.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        plt.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
        plt.tight_layout()

        output_path = os.path.join(figures_dir, "all_envs_methods_avg_eval_return.png")
        plt.savefig(output_path, dpi=800)
        print(f"Saved plot to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())