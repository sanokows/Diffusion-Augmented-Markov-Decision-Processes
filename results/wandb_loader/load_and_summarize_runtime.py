#!/usr/bin/env python3

import argparse
from collections import defaultdict
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import wandb


PROJECT_SUFFIXES = ["_FR_16_01", "_FR_19_01", "_FR_24_01", "_FR_30_01", "_FR_test_PPO", "_FR_ME-WPO", "_FR_WPO"]
ENV_NAMES = [
    "FingerSpin",
    "AcrobotSwingup",
    "PendulumSwingup",
    "CartpoleSwingupSparse",
    "AcrobotSwingupSparse",
    "CheetahRun",
    "FishSwim",
    "HopperHop",
    "HopperStand",
    "WalkerStand",
    "WalkerRun",
    "WalkerWalk",
]
RUNS_BY_SUFFIX: dict[str, dict[str, str | dict[str, str]]] = {
    "_FR_16_01": {
        "reppo-dime-debug-1-<env_name>": {"alias": "REPPO-DiME", "color": "#ff0000"},
    },
    "_FR_19_01": {
        "reppo-<env_name>-reparam": {"alias": "REPPO", "color": "#8b1a1a"},
    },
    "_FR_24_01": {
        "reppo-dmerl-debug-1-<env_name>-WPO": {
            "alias": "DME-WPO (ours)",
            "color": "#1b7f3a",
        },
    },
    "_FR_30_01": {
        "reppo-dmerl-debug-1-<env_name>-reparam": {
            "alias": "DME-REPPO (ours)",
            "color": "#8b1a1a",
        },
    },
    "_FR_test_PPO": {
        "ppo-diff_ppo-<env_name>": {"alias": "DME-PPO (ours)", "color": "#3b528b"},
    },
    "_FR_ME-WPO": {
        "reppo-<env_name>-WPO": {"alias": "ME-WPO (ours)", "color": "#1b7f3a"},
    },
    "_FR_WPO": {
        "reppo-<env_name>-WPO": {"alias": "WPO", "color": "#9bd65a"},
    },
}

RUNTIME_KEYS = [
    "_runtime",
    "runtime",
    "wall_time",
    "train/runtime",
    "train_runtime",
    "train/time",
    "time/elapsed",
    "elapsed_time",
    "elapsed",
]
CRITIC_PARAM_KEYS = ["norm/init/critic_params", "norm_init/critic_params"]
CRITIC_NORM_KEYS = ["norm_init/critic", "norm/init/critic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load W&B runs, group by method (from run name), and report average "
            "runtime and critic init stats per environment and across environments."
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
        "--runtime-key",
        default="auto",
        help=(
            "Metric key for runtime in seconds. Use 'auto' to try common keys "
            f"({', '.join(RUNTIME_KEYS)})."
        ),
    )
    parser.add_argument(
        "--time-unit",
        choices=["seconds", "minutes", "hours"],
        default="hours",
        help="Unit for reporting runtime averages.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Optional CSV output path for per-env summary. If provided, an "
            "additional *_overall.csv file is written for cross-env averages."
        ),
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
        help="Print details about missing metrics for runs.",
    )
    return parser.parse_args()


def resolve_project_path(api: wandb.Api, entity: str | None, project: str) -> str:
    if entity:
        return f"{entity}/{project}"
    default_entity = getattr(api, "default_entity", None)
    if default_entity:
        return f"{default_entity}/{project}"
    return project


def resolve_project_suffix(project: str) -> str | None:
    for suffix in PROJECT_SUFFIXES:
        if project.endswith(suffix):
            return suffix
    return None


def normalize_runs_by_suffix(
    mapping: dict[str, dict[str, str | dict[str, str]]] | None,
) -> dict[str, dict[str, dict[str, str]]]:
    if not mapping:
        return {}
    normalized: dict[str, dict[str, dict[str, str]]] = {}
    for suffix, run_map in mapping.items():
        if not suffix or not run_map:
            continue
        clean_suffix = suffix.strip()
        if not clean_suffix:
            continue
        normalized_runs: dict[str, dict[str, str]] = {}
        for run_name, spec in run_map.items():
            if not run_name or not run_name.strip():
                continue
            run_key = run_name.strip()
            if isinstance(spec, dict):
                alias = (spec.get("alias") or run_key).strip()
                color = (spec.get("color") or "").strip()
            else:
                alias = str(spec).strip() if spec is not None else run_key
                color = ""
            normalized_runs[run_key] = {"alias": alias or run_key, "color": color}
        if normalized_runs:
            normalized[clean_suffix] = normalized_runs
    return normalized


def expand_name_map(
    name_map: dict[str, dict[str, str]], env_name: str
) -> dict[str, dict[str, str]]:
    expanded: dict[str, dict[str, str]] = {}
    env_lower = env_name.lower()
    for raw_name, spec in name_map.items():
        if "<env_name>" in raw_name:
            expanded[raw_name.replace("<env_name>", env_lower)] = spec
        else:
            expanded[raw_name] = spec
    return expanded


def match_run_name(
    raw_name: str, name_map: dict[str, dict[str, str]]
) -> tuple[dict[str, str] | None, str | None]:
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


def infer_env_from_project(project: str) -> str:
    for env_name in ENV_NAMES:
        if env_name.lower() in project.lower():
            return env_name
    return project


def _summary_dict(run: wandb.apis.public.Run) -> dict[str, Any]:
    try:
        return dict(run.summary or {})
    except Exception:
        return {}


def _nested_lookup(summary: dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary[key]
    if "/" not in key:
        return None
    cur: Any = summary
    for part in key.split("/"):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if math.isnan(val):
        return None
    return val


def extract_runtime_seconds(summary: dict[str, Any], runtime_key: str) -> float | None:
    if runtime_key != "auto":
        return _coerce_float(_nested_lookup(summary, runtime_key))
    for key in RUNTIME_KEYS:
        val = _coerce_float(_nested_lookup(summary, key))
        if val is not None:
            return val
    start_time = _coerce_float(_nested_lookup(summary, "_start_time"))
    end_time = _coerce_float(_nested_lookup(summary, "_timestamp"))
    if start_time is not None and end_time is not None and end_time >= start_time:
        return end_time - start_time
    return None


def extract_first(summary: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        val = _coerce_float(_nested_lookup(summary, key))
        if val is not None:
            return val
    return None


def seconds_to_unit(seconds: float | None, unit: str) -> float | None:
    if seconds is None:
        return None
    if unit == "seconds":
        return seconds
    if unit == "minutes":
        return seconds / 60.0
    return seconds / 3600.0


def summarize_group(df: pd.DataFrame, time_unit: str, baseline_method: str) -> pd.DataFrame:
    def mean_or_nan(series: pd.Series) -> float:
        series = series.dropna()
        if series.empty:
            return float("nan")
        return float(series.mean())

    def std_or_nan(series: pd.Series) -> float:
        series = series.dropna()
        if series.empty:
            return float("nan")
        return float(series.std())

    def min_or_nan(series: pd.Series) -> float:
        series = series.dropna()
        if series.empty:
            return float("nan")
        return float(series.min())

    def max_or_nan(series: pd.Series) -> float:
        series = series.dropna()
        if series.empty:
            return float("nan")
        return float(series.max())

    agg = df.groupby(["env", "method"]).agg(
        run_count=("run_id", "count"),
        runtime_mean_s=("runtime_s", mean_or_nan),
        runtime_std_s=("runtime_s", std_or_nan),
        runtime_min_s=("runtime_s", min_or_nan),
        runtime_max_s=("runtime_s", max_or_nan),
        runtime_count=("runtime_s", "count"),
        critic_params_mean=("critic_params", mean_or_nan),
        critic_params_count=("critic_params", "count"),
        critic_norm_mean=("critic_norm", mean_or_nan),
        critic_norm_count=("critic_norm", "count"),
    )
    agg = agg.reset_index()
    agg["runtime_mean"] = agg["runtime_mean_s"].apply(lambda v: seconds_to_unit(v, time_unit))
    agg["runtime_std"] = agg["runtime_std_s"].apply(lambda v: seconds_to_unit(v, time_unit))
    agg["runtime_min"] = agg["runtime_min_s"].apply(lambda v: seconds_to_unit(v, time_unit))
    agg["runtime_max"] = agg["runtime_max_s"].apply(lambda v: seconds_to_unit(v, time_unit))
    baseline_map = (
        agg.loc[agg["method"] == baseline_method, ["env", "runtime_mean"]]
        .set_index("env")["runtime_mean"]
        .to_dict()
    )
    agg["factor_vs_reppo"] = agg.apply(
        lambda row: row["runtime_mean"] / baseline_map.get(row["env"], float("nan"))
        if pd.notna(row["runtime_mean"]) and baseline_map.get(row["env"]) not in (None, 0, float("nan"))
        else float("nan"),
        axis=1,
    )
    return agg


def summarize_overall(
    env_summary: pd.DataFrame, time_unit: str, baseline_method: str
) -> pd.DataFrame:
    def mean_or_nan(series: pd.Series) -> float:
        series = series.dropna()
        if series.empty:
            return float("nan")
        return float(series.mean())

    env_means = env_summary.copy()
    overall = env_means.groupby("method").agg(
        env_count=("env", "nunique"),
        runtime_mean=("runtime_mean", mean_or_nan),
        critic_params_mean=("critic_params_mean", mean_or_nan),
        critic_norm_mean=("critic_norm_mean", mean_or_nan),
        factor_vs_reppo=("factor_vs_reppo", mean_or_nan),
    )
    overall = overall.reset_index()
    overall["time_unit"] = time_unit
    return overall


def format_latex_runtime_table(
    overall_summary: pd.DataFrame,
    time_unit: str,
) -> str:
    table = overall_summary.copy()
    table = table[["method", "runtime_mean", "factor_vs_reppo", "env_count"]]
    table = table.sort_values(["runtime_mean", "method"], na_position="last")
    table = table.rename(
        columns={
            "method": "Method",
            "runtime_mean": f"Runtime ({time_unit})",
            "factor_vs_reppo": "Factor vs REPPO",
            "env_count": "Env count",
        }
    )
    table[f"Runtime ({time_unit})"] = table[f"Runtime ({time_unit})"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "--"
    )
    table["Factor vs REPPO"] = table["Factor vs REPPO"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "--"
    )
    return table.to_latex(index=False, escape=False)


def print_env_runtime_breakdown(env_summary: pd.DataFrame, time_unit: str) -> None:
    runtime_cols = [
        "env",
        "method",
        "runtime_mean",
        "runtime_std",
        "runtime_min",
        "runtime_max",
        "runtime_count",
    ]
    runtime_view = env_summary[runtime_cols].copy()
    runtime_view = runtime_view.sort_values(["env", "method"])

    print(f"\nAverage runtime by env and method ({time_unit})")
    for env_name, group in runtime_view.groupby("env", sort=True):
        print(f"{env_name}:")
        for _, row in group.iterrows():
            runtime_mean = row["runtime_mean"]
            runtime_std = row["runtime_std"]
            runtime_min = row["runtime_min"]
            runtime_max = row["runtime_max"]
            mean_str = f"{runtime_mean:.3f}" if pd.notna(runtime_mean) else "--"
            std_str = f"{runtime_std:.3f}" if pd.notna(runtime_std) else "--"
            min_str = f"{runtime_min:.3f}" if pd.notna(runtime_min) else "--"
            max_str = f"{runtime_max:.3f}" if pd.notna(runtime_max) else "--"
            print(
                f"  {row['method']}: mean={mean_str}, std={std_str}, min={min_str}, max={max_str} "
                f"(n={int(row['runtime_count'])} runs with runtime)"
            )


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

    filters = {"state": args.state} if args.state else {}
    records: list[dict[str, Any]] = []

    for env_name, projects in env_projects.items():
        skipped = 0
        for project in projects:
            project_path = resolve_project_path(api, args.entity, project)
            project_suffix = resolve_project_suffix(project)
            suffix_run_map = runs_by_suffix.get(project_suffix) if project_suffix else None
            expanded_suffix_map = (
                expand_name_map(suffix_run_map, env_name) if suffix_run_map else None
            )
            found_suffix_runs: set[str] = set()
            if expanded_suffix_map is not None:
                wanted_names = ", ".join(sorted(expanded_suffix_map.keys()))
                print(f"{env_name} - {project_path}: wanted run names: {wanted_names}")
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
                    run_spec, matched_key = match_run_name(raw_name, expanded_suffix_map)
                    if run_spec is None:
                        continue
                    alias = run_spec["alias"]
                    print(
                        f"{env_name} - loaded run: {raw_name} "
                        f"(alias: {alias}, matched: {matched_key})"
                    )

                summary = _summary_dict(run)
                runtime_s = extract_runtime_seconds(summary, args.runtime_key)
                critic_params = extract_first(summary, CRITIC_PARAM_KEYS)
                critic_norm = extract_first(summary, CRITIC_NORM_KEYS)

                if expanded_suffix_map is None:
                    method_name = clean_method_name(raw_name)
                else:
                    method_name = alias
                    found_suffix_runs.add(matched_key or raw_name)

                if runtime_s is None and critic_params is None and critic_norm is None:
                    skipped += 1
                    if args.verbose:
                        print(
                            f"Warning: no usable runtime/critic stats for {raw_name}",
                            file=sys.stderr,
                        )
                    continue

                records.append(
                    {
                        "env": env_name,
                        "project": project,
                        "method": method_name,
                        "run_id": run.id,
                        "run_name": raw_name,
                        "runtime_s": runtime_s,
                        "critic_params": critic_params,
                        "critic_norm": critic_norm,
                    }
                )

            if expanded_suffix_map is not None:
                missing = set(expanded_suffix_map.keys()) - found_suffix_runs
                if missing:
                    missing_list = ", ".join(sorted(missing))
                    print(
                        f"Warning: missing specified runs in {project_path}: {missing_list}",
                        file=sys.stderr,
                    )

        if skipped:
            print(f"Skipped {skipped} runs without usable stats in {env_name}.")

    if not records:
        print("No runs contained runtime/critic stats.", file=sys.stderr)
        return 1

    df = pd.DataFrame.from_records(records)
    baseline_method = "REPPO"
    env_summary = summarize_group(df, args.time_unit, baseline_method)
    overall_summary = summarize_overall(env_summary, args.time_unit, baseline_method)

    print("\nPer-environment averages")
    print(env_summary.sort_values(["env", "method"]).to_string(index=False))

    if multi_env:
        print("\nAcross-env averages (mean of per-env means)")
        print(overall_summary.sort_values(["method"]).to_string(index=False))
        print("\nLaTeX table (runtime comparison across envs)")
        print(format_latex_runtime_table(overall_summary, args.time_unit))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        env_summary.to_csv(out_path, index=False)
        base = out_path.with_suffix("")
        overall_path = base.with_name(base.name + "_overall").with_suffix(".csv")
        overall_summary.to_csv(overall_path, index=False)
        print(f"Wrote {out_path}")
        if multi_env:
            print(f"Wrote {overall_path}")

    print_env_runtime_breakdown(env_summary, args.time_unit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
