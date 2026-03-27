#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from timing_wandb_utils import (
    METHOD_ORDER,
    collect_timing_data,
    resolve_projects,
    split_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load timing runs from W&B and export cleaned timing series plus per-run summary."
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
        help="Output directory for CSV exports.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skipped runs and fetch issues.",
    )
    return parser.parse_args()


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

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    series_csv = out_dir / "timing_series_cleaned.csv"
    runs_csv = out_dir / "timing_run_summary.csv"
    collected.series_df.to_csv(series_csv, index=False)
    if not collected.run_summary_df.empty:
        collected.run_summary_df["method_order"] = collected.run_summary_df["method"].map(
            {name: idx for idx, name in enumerate(METHOD_ORDER)}
        ).fillna(1e9)
        collected.run_summary_df = collected.run_summary_df.sort_values(
            ["env_name", "method_order", "run_name"]
        ).drop(columns=["method_order"])
    collected.run_summary_df.to_csv(runs_csv, index=False)

    print(f"[summary] projects={resolved_projects}")
    print(f"[summary] exported series rows: {len(collected.series_df)} -> {series_csv}")
    print(f"[summary] exported run rows: {len(collected.run_summary_df)} -> {runs_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
