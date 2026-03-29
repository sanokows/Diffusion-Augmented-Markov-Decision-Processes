#!/usr/bin/env python3
"""
This script produces two families of plots for each method at each training step:

1) Mean plots:
   - Central curve: arithmetic mean over all available run values in a (method, step) group.
   - Uncertainty band: standard error of the mean (SEM) over the same full set of runs.
   - Interpretation: best when you want the conventional average performance and every seed
     should contribute equally, including extreme high/low outcomes.

2) IQM plots (Interquartile Mean):
   - Central curve: values in each (method, step) group are sorted; only the middle 50%
     (25th-75th percentile slice) is averaged.
   - Uncertainty band: SEM computed on that same middle-50% subset.
   - Interpretation: more robust to outlier seeds, because unusually bad/good runs in the
     outer quartiles do not affect the IQM center or its error band.

In short: mean plots summarize all seeds directly, while IQM plots summarize the robust
"typical" seed behavior by trimming the outer quartiles before averaging.
"""

import argparse
from collections import defaultdict
import math
import os
import re
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import wandb


DEFAULT_Y_KEY = "eval/episode_return"
AUTO_X_KEYS = ["_step"]
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
# Optional per-suffix run selection with aliases and colors.
# Format:
# {
#   "<suffix>": {
#     "<run_name>": {"alias": "<alias>", "color": "<hex>"},
#     "<run_name>": "<alias>",  # legacy form (no color)
#   },
# }
# Leave empty or omit suffix keys to use default name-based grouping.
#RUNS_BY_SUFFIX: dict[str, dict[str, str]] = {}
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
PPO_BRAX_LABEL = "PPO (r)"
CSV_RESULTS_DIR = Path("results")
PLOT_MAX_STEPS = 5e7
VIRIDIS_RANGE = (0.1, 0.95)
LINE_STYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 2))]
LINE_WIDTH = 4.0
EXTRAPOLATE_ENV_NAMES = {"WalkerWalk", "WalkerRun", "WalkerStand"}
EXTRAPOLATE_MIN_FRACTION = 0.8
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 17
LEGEND_FONTSIZE = 18
LEGEND_FONTSIZE_all = 14
LEGEND_HANDLELENGTH = 4.0
TITLE_FONTSIZE = 22
GRID_ALPHA = 0.9
alpha = 0.1
alpha_line = 0.7
GRID_LINESTYLE = ":"
GRID_LINEWIDTH = 2.0
GRID_COLS = 3
METHOD_COLOR_OVERRIDES = {
    PPO_BRAX_LABEL: "#414487",
}
METHOD_STYLE_OVERRIDES = {
    PPO_BRAX_LABEL: "--",
    "DME-PPO (ours)": "-",
    "ME-WPO (ours)": "--",
    "DME-WPO (ours)": "-",
    "REPPO": "--",
    "DME-REPPO (ours)": "-",
    "REPPO-DiME": "-",
}
STYLE_VERSION_LEGACY = "legacy"
STYLE_VERSION_REVIEWER = "reviewer"
REVIEWER_STYLE_CHOICES = [STYLE_VERSION_LEGACY, STYLE_VERSION_REVIEWER]
REVIEWER_LINESTYLE_BY_CATEGORY = {
    "proposed": "-",
    "paired_baseline": "--",
    "other_baseline": ":",
}
REVIEWER_CATEGORY_ORDER = {
    "proposed": 0,
    "paired_baseline": 1,
    "other_baseline": 2,
}
REVIEWER_PROPOSED_TO_BASELINES = {
    "dme-ppo": {"ppo (r)", "ppo_brax", "ppo"},
    "dme-reppo": {"reppo"},
    "dme-wpo": {"wpo", "me-wpo"},
}
REVIEWER_EXTRA_PROPOSED = {"reppo-dime"}
REVIEWER_SPECIAL_LINESTYLE = (0, (7, 2.2, 1.8, 2.2))
REVIEWER_METHOD_STYLE_OVERRIDES = {
    "wpo": REVIEWER_SPECIAL_LINESTYLE,
    "reppo-dime": REVIEWER_SPECIAL_LINESTYLE,
}
REVIEWER_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#332288",
    "#117733",
    "#AA4499",
    "#44AA99",
]
MEAN_FIGURES_SUBDIR = "mean"
IQM_FIGURES_SUBDIR = "IQM"


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
    parser.add_argument(
        "--grid-alpha",
        type=float,
        default=GRID_ALPHA,
        help="Gridline alpha for plots.",
    )
    parser.add_argument(
        "--grid-linewidth",
        type=float,
        default=GRID_LINEWIDTH,
        help="Gridline linewidth for plots.",
    )
    parser.add_argument(
        "--style-version",
        choices=REVIEWER_STYLE_CHOICES,
        default=STYLE_VERSION_REVIEWER,
        help=(
            "Plot style preset. 'legacy' keeps the original color/style settings; "
            "'reviewer' uses higher-contrast colors and 3 line-style categories."
        ),
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


def canonical_method_name(method_name: str) -> str:
    canonical = method_name.lower().strip()
    canonical = canonical.replace("(ours)", "").strip()
    canonical = re.sub(r"\s+", " ", canonical)
    return canonical


def build_reviewer_categories(methods: set[str]) -> dict[str, str]:
    canonical_by_method = {method: canonical_method_name(method) for method in methods}
    methods_by_canonical: dict[str, set[str]] = defaultdict(set)
    for method, canonical in canonical_by_method.items():
        methods_by_canonical[canonical].add(method)

    proposed_methods = {
        method
        for method in methods
        if "(ours)" in method.lower()
        or canonical_by_method[method].startswith("dme-")
        or canonical_by_method[method] in REVIEWER_EXTRA_PROPOSED
    }

    paired_baselines: set[str] = set()
    for proposed in proposed_methods:
        proposed_canonical = canonical_by_method[proposed]
        baseline_canonicals = REVIEWER_PROPOSED_TO_BASELINES.get(proposed_canonical, set())
        for baseline_canonical in baseline_canonicals:
            for baseline in methods_by_canonical.get(baseline_canonical, set()):
                if baseline not in proposed_methods:
                    paired_baselines.add(baseline)

    categories = {method: "other_baseline" for method in methods}
    for method in proposed_methods:
        categories[method] = "proposed"
    for method in paired_baselines:
        categories[method] = "paired_baseline"
    return categories


def build_reviewer_style_maps(
    methods: set[str],
) -> tuple[dict[str, str], dict[str, object], dict[str, str]]:
    categories = build_reviewer_categories(methods)
    canonical_by_method = {method: canonical_method_name(method) for method in methods}
    colors: dict[str, str] = {}
    styles: dict[str, object] = {}
    color_index = 0

    def next_color() -> str:
        nonlocal color_index
        color = REVIEWER_PALETTE[color_index % len(REVIEWER_PALETTE)]
        color_index += 1
        return color

    proposed_methods = sorted(
        [method for method in methods if categories.get(method) == "proposed"],
        key=canonical_method_name,
    )
    for method in proposed_methods:
        colors[method] = next_color()

    for method in sorted(
        [m for m in methods if categories.get(m) == "paired_baseline"],
        key=canonical_method_name,
    ):
        baseline_canonical = canonical_by_method[method]
        matched_color = None
        for proposed in proposed_methods:
            proposed_canonical = canonical_by_method[proposed]
            candidate_baselines = REVIEWER_PROPOSED_TO_BASELINES.get(
                proposed_canonical, set()
            )
            if baseline_canonical in candidate_baselines:
                matched_color = colors.get(proposed)
                break
        colors[method] = matched_color or next_color()

    for method in sorted(
        [m for m in methods if m not in colors], key=canonical_method_name
    ):
        colors[method] = next_color()

    for method in methods:
        category = categories.get(method, "other_baseline")
        styles[method] = REVIEWER_LINESTYLE_BY_CATEGORY.get(category, ":")
        canonical = canonical_by_method[method]
        if canonical in REVIEWER_METHOD_STYLE_OVERRIDES:
            styles[method] = REVIEWER_METHOD_STYLE_OVERRIDES[canonical]

    return colors, styles, categories


def build_method_order(methods: set[str], categories: dict[str, str]) -> list[str]:
    return sorted(
        methods,
        key=lambda method: (
            REVIEWER_CATEGORY_ORDER.get(categories.get(method, "other_baseline"), 99),
            canonical_method_name(method),
        ),
    )


def ordered_methods_present(df: pd.DataFrame, method_order: list[str]) -> list[str]:
    present = set(df["method"].unique())
    return [method for method in method_order if method in present]


def make_legend_clearer(legend: object) -> None:
    if legend is None:
        return
    for line in legend.get_lines():
        line.set_alpha(1.0)


def build_distinct_palette(count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return [mcolors.to_hex(cm.viridis(0.6))]
    start, end = VIRIDIS_RANGE
    step = (end - start) / (count - 1)
    return [mcolors.to_hex(cm.viridis(start + step * idx)) for idx in range(count)]


def interquartile_mean_and_stderr(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().sort_values().to_numpy()
    n = clean.size
    if n == 0:
        return float("nan"), float("nan")
    lower = int(math.floor(0.25 * n))
    upper = int(math.ceil(0.75 * n))
    if upper <= lower:
        middle = clean
    else:
        middle = clean[lower:upper]
    if middle.size == 0:
        middle = clean
    middle_series = pd.Series(middle)
    iqm = float(middle_series.mean())
    stderr = float(middle_series.sem()) if middle_series.size > 1 else float("nan")
    return iqm, stderr


def aggregate_method_step(data: pd.DataFrame, use_iqm: bool) -> pd.DataFrame:
    grouped = data.groupby(["method", "step"])["value"]
    if use_iqm:
        iqm = grouped.apply(
            lambda values: pd.Series(
                interquartile_mean_and_stderr(values), index=["value", "stderr"]
            )
        ).unstack().reset_index()
        return iqm
    mean = grouped.mean().reset_index()
    stderr = grouped.sem().reset_index().rename(columns={"value": "stderr"})
    return mean.merge(stderr, on=["method", "step"], how="left")


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
    grid_alpha = args.grid_alpha
    grid_linewidth = args.grid_linewidth

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
    style_dir_suffix = (
        ""
        if args.style_version == STYLE_VERSION_LEGACY
        else f"_{args.style_version}"
    )
    mean_figures_dir = os.path.join(figures_dir, f"{MEAN_FIGURES_SUBDIR}{style_dir_suffix}")
    iqm_figures_dir = os.path.join(figures_dir, f"{IQM_FIGURES_SUBDIR}{style_dir_suffix}")
    os.makedirs(mean_figures_dir, exist_ok=True)
    os.makedirs(iqm_figures_dir, exist_ok=True)
    out_stem: str | None = None
    out_ext = ""
    if args.out:
        out_name = os.path.basename(args.out)
        out_stem, out_ext = os.path.splitext(out_name)
    filters = {"state": args.state} if args.state else {}

    env_results = []
    all_methods = set()
    method_color_overrides: dict[str, str] = dict(METHOD_COLOR_OVERRIDES)
    method_style_overrides: dict[str, str] = dict(METHOD_STYLE_OVERRIDES)


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
                    run_spec, matched_key = match_run_name(raw_name, expanded_suffix_map)
                    if run_spec is None:
                        continue
                    alias = run_spec["alias"]
                    color_override = run_spec.get("color", "")
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
                    if color_override:
                        method_color_overrides[method_name] = color_override
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

    if args.style_version == STYLE_VERSION_REVIEWER:
        color_by_method, style_by_method, method_categories = build_reviewer_style_maps(
            all_methods
        )
    else:
        color_by_method = {}
        style_by_method = {}
        method_categories = {method: "legacy" for method in all_methods}
        palette = build_distinct_palette(len(all_methods))
        for idx, method in enumerate(sorted(all_methods)):
            color_by_method[method] = palette[idx % len(palette)]
            style_by_method[method] = LINE_STYLES[idx % len(LINE_STYLES)]
        for method, color in method_color_overrides.items():
            color_by_method[method] = color
        for method, style in method_style_overrides.items():
            style_by_method[method] = style

    method_order = build_method_order(all_methods, method_categories)

    for result in env_results:
        env_name = result["env_name"]
        data = result["records"]
        method_run_counts = result["method_run_counts"]
        skipped = result["skipped"]

        merged = aggregate_method_step(data, use_iqm=False)
        merged_iqm = aggregate_method_step(data, use_iqm=True)

        result["merged"] = merged
        result["merged_iqm"] = merged_iqm

        plt.figure(figsize=(9, 5))
        for method in ordered_methods_present(merged, method_order):
            method_df = merged[merged["method"] == method].sort_values("step")
            run_count = len(method_run_counts[method])
            label = f"{method}"# (n={run_count})"
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

        plt.xlabel("env calls", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("episode return", fontsize=AXIS_LABEL_FONTSIZE)
        plt.title(f"{env_name}", fontsize = TITLE_FONTSIZE)
        legend = plt.legend(
            loc="lower right",
            ncol=2,
            fontsize=LEGEND_FONTSIZE,
            handlelength=LEGEND_HANDLELENGTH,
        )
        make_legend_clearer(legend)
        plt.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        plt.grid(True, linestyle=GRID_LINESTYLE, alpha=grid_alpha, linewidth=grid_linewidth)
        plt.xlim(left=0, right=PLOT_MAX_STEPS)
        plt.tight_layout()

        if out_stem is not None:
            if multi_env:
                mean_filename = f"{out_stem}_{env_name}{out_ext}"
            else:
                mean_filename = f"{out_stem}{out_ext}"
        else:
            mean_filename = f"dime_{env_name}_methods_avg_eval_return.png"
        output_path = os.path.join(mean_figures_dir, mean_filename)

        plt.savefig(output_path, dpi=800)
        print(f"Saved plot to {output_path}")
        plt.close()

        plt.figure(figsize=(9, 5))
        for method in ordered_methods_present(merged_iqm, method_order):
            method_df = merged_iqm[merged_iqm["method"] == method].sort_values("step")
            run_count = len(method_run_counts[method])
            label = f"{method}"# (n={run_count})"
            color = color_by_method.get(method)
            linestyle = style_by_method.get(method, "-")
            plt.plot(
                method_df["step"],
                method_df["value"],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=LINE_WIDTH,
                alpha=alpha_line,
            )
            if method_df["stderr"].notna().any():
                plt.fill_between(
                    method_df["step"],
                    method_df["value"] - method_df["stderr"],
                    method_df["value"] + method_df["stderr"],
                    color=color,
                    alpha=alpha,
                )

        plt.xlabel("env calls", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("episode return (IQM)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.title(env_name, fontsize=TITLE_FONTSIZE)
        legend = plt.legend(
            loc="lower right",
            ncol=2,
            fontsize=LEGEND_FONTSIZE,
            handlelength=LEGEND_HANDLELENGTH,
        )
        make_legend_clearer(legend)
        plt.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        plt.grid(True, linestyle=GRID_LINESTYLE, alpha=grid_alpha, linewidth=grid_linewidth)
        plt.xlim(left=0, right=PLOT_MAX_STEPS)
        plt.tight_layout()

        if out_stem is not None:
            if multi_env:
                iqm_filename = f"{out_stem}_{env_name}_iqm{out_ext}"
            else:
                iqm_filename = f"{out_stem}_iqm{out_ext}"
        else:
            iqm_filename = f"dime_{env_name}_methods_iqm_eval_return.png"
        output_iqm_path = os.path.join(iqm_figures_dir, iqm_filename)
        plt.savefig(output_iqm_path, dpi=800)
        print(f"Saved IQM plot to {output_iqm_path}")
        plt.close()

        if skipped:
            print(f"Skipped {skipped} runs without usable data in {env_name}.")

    if multi_env and env_results:
        num_envs = len(env_results)
        ncols = min(GRID_COLS, num_envs)
        nrows = math.ceil(num_envs / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 4.2, nrows * 3.2),
            sharex=True,
            sharey=False,
        )
        if isinstance(axes, Axes):
            axes_list = [axes]
        else:
            axes_list = list(axes.ravel())
        legend_handles: dict[str, plt.Line2D] = {}

        for idx, result in enumerate(env_results):
            ax = axes_list[idx]
            env_name = result["env_name"]
            merged = result["merged"]
            method_run_counts = result["method_run_counts"]

            for method in ordered_methods_present(merged, method_order):
                method_df = merged[merged["method"] == method].sort_values("step")
                run_count = len(method_run_counts[method])
                label = f"{method}"# (n={run_count})"
                color = color_by_method.get(method)
                linestyle = style_by_method.get(method, "-")
                (line,) = ax.plot(
                    method_df["step"],
                    method_df["value"],
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=LINE_WIDTH,
                    alpha=alpha_line,
                )
                if method not in legend_handles:
                    legend_handles[method] = line
                if method_df["stderr"].notna().any():
                    ax.fill_between(
                        method_df["step"],
                        method_df["value"] - method_df["stderr"],
                        method_df["value"] + method_df["stderr"],
                        color=color,
                        alpha=alpha,
                    )

            ax.set_title(env_name, fontsize=TITLE_FONTSIZE)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(True, linestyle=GRID_LINESTYLE, alpha=grid_alpha, linewidth=grid_linewidth)
            ax.set_xlim(left=0, right=PLOT_MAX_STEPS)

        for idx in range(num_envs, len(axes_list)):
            fig.delaxes(axes_list[idx])
        fig.supxlabel("env calls", fontsize=AXIS_LABEL_FONTSIZE)
        fig.supylabel("episode return", fontsize=AXIS_LABEL_FONTSIZE)
        if legend_handles:
            handles = [
                legend_handles[m] for m in method_order if m in legend_handles
            ]
            labels = [h.get_label() for h in handles]
            legend_cols = max(1, math.ceil(len(handles) / 2))
            legend = fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=legend_cols,
                fontsize=LEGEND_FONTSIZE,
                handlelength=LEGEND_HANDLELENGTH,
            )
            make_legend_clearer(legend)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        if out_stem is not None:
            grid_output = os.path.join(mean_figures_dir, f"{out_stem}_grid{out_ext}")
        else:
            grid_output = os.path.join(
                mean_figures_dir, "all_envs_methods_grid_eval_return.png"
            )
        fig.savefig(grid_output, dpi=800, bbox_inches="tight")
        print(f"Saved plot to {grid_output}")
        plt.close(fig)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 4.2, nrows * 3.2),
            sharex=True,
            sharey=False,
        )
        if isinstance(axes, Axes):
            axes_list = [axes]
        else:
            axes_list = list(axes.ravel())
        legend_handles = {}

        for idx, result in enumerate(env_results):
            ax = axes_list[idx]
            env_name = result["env_name"]
            merged_iqm = result["merged_iqm"]
            method_run_counts = result["method_run_counts"]

            for method in ordered_methods_present(merged_iqm, method_order):
                method_df = merged_iqm[merged_iqm["method"] == method].sort_values("step")
                run_count = len(method_run_counts[method])
                label = f"{method}"# (n={run_count})"
                color = color_by_method.get(method)
                linestyle = style_by_method.get(method, "-")
                (line,) = ax.plot(
                    method_df["step"],
                    method_df["value"],
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=LINE_WIDTH,
                    alpha=alpha_line,
                )
                if method not in legend_handles:
                    legend_handles[method] = line
                if method_df["stderr"].notna().any():
                    ax.fill_between(
                        method_df["step"],
                        method_df["value"] - method_df["stderr"],
                        method_df["value"] + method_df["stderr"],
                        color=color,
                        alpha=alpha,
                    )

            ax.set_title(env_name, fontsize=TITLE_FONTSIZE)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(True, linestyle=GRID_LINESTYLE, alpha=grid_alpha, linewidth=grid_linewidth)
            ax.set_xlim(left=0, right=PLOT_MAX_STEPS)

        for idx in range(num_envs, len(axes_list)):
            fig.delaxes(axes_list[idx])
        fig.supxlabel("env calls", fontsize=AXIS_LABEL_FONTSIZE)
        fig.supylabel("episode return (IQM)", fontsize=AXIS_LABEL_FONTSIZE)
        if legend_handles:
            handles = [
                legend_handles[m] for m in method_order if m in legend_handles
            ]
            labels = [h.get_label() for h in handles]
            legend_cols = max(1, math.ceil(len(handles) / 2))
            legend = fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=legend_cols,
                fontsize=LEGEND_FONTSIZE,
                handlelength=LEGEND_HANDLELENGTH,
            )
            make_legend_clearer(legend)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        if out_stem is not None:
            grid_iqm_output = os.path.join(iqm_figures_dir, f"{out_stem}_grid_iqm{out_ext}")
        else:
            grid_iqm_output = os.path.join(
                iqm_figures_dir, "all_envs_methods_grid_iqm_eval_return.png"
            )
        fig.savefig(grid_iqm_output, dpi=800, bbox_inches="tight")
        print(f"Saved IQM plot to {grid_iqm_output}")
        plt.close(fig)

    if multi_env and env_results:
        overall_records = pd.concat([r["records"] for r in env_results], ignore_index=True)
        overall_merged = aggregate_method_step(overall_records, use_iqm=False)
        overall_merged_iqm = aggregate_method_step(overall_records, use_iqm=True)

        overall_method_runs: dict[str, set[str]] = defaultdict(set)
        for result in env_results:
            for method, runs in result["method_run_counts"].items():
                overall_method_runs[method].update(runs)
        overall_run_counts = {method: len(runs) for method, runs in overall_method_runs.items()}

        plt.figure(figsize=(9, 5))
        for method in ordered_methods_present(overall_merged, method_order):
            method_df = overall_merged[overall_merged["method"] == method].sort_values("step")
            run_count = overall_run_counts.get(method, 0)
            label = f"{method}"# (n={run_count})"
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

        plt.xlabel("env calls", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("episode return", fontsize=AXIS_LABEL_FONTSIZE)
        plt.title(f"All environments", fontsize=TITLE_FONTSIZE)
        legend = plt.legend(
            loc="lower right",
            ncol=2,
            fontsize=LEGEND_FONTSIZE_all,
            handlelength=LEGEND_HANDLELENGTH,
        )
        make_legend_clearer(legend)
        plt.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        plt.grid(True, linestyle=GRID_LINESTYLE, alpha=grid_alpha, linewidth=grid_linewidth)
        plt.tight_layout()

        output_path = os.path.join(mean_figures_dir, "all_envs_methods_avg_eval_return.png")
        plt.savefig(output_path, dpi=800)
        print(f"Saved plot to {output_path}")
        plt.close()

        plt.figure(figsize=(9, 5))
        for method in ordered_methods_present(overall_merged_iqm, method_order):
            method_df = overall_merged_iqm[overall_merged_iqm["method"] == method].sort_values(
                "step"
            )
            run_count = overall_run_counts.get(method, 0)
            label = f"{method}"# (n={run_count})"
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

        plt.xlabel("env calls", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("episode return (IQM)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.title("All environments", fontsize=TITLE_FONTSIZE)
        legend = plt.legend(
            loc="lower right",
            ncol=2,
            fontsize=LEGEND_FONTSIZE_all,
            handlelength=LEGEND_HANDLELENGTH,
        )
        make_legend_clearer(legend)
        plt.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        plt.grid(True, linestyle=GRID_LINESTYLE, alpha=grid_alpha, linewidth=grid_linewidth)
        plt.tight_layout()

        output_iqm_path = os.path.join(iqm_figures_dir, "all_envs_methods_iqm_eval_return.png")
        plt.savefig(output_iqm_path, dpi=800)
        print(f"Saved IQM plot to {output_iqm_path}")
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
