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
from matplotlib.lines import Line2D
import pandas as pd
import wandb
try:
    from load_and_summarize_runtime import RUNTIME_KEYS, extract_runtime_seconds
except Exception:
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

    def extract_runtime_seconds(summary: dict[str, object], runtime_key: str) -> float | None:
        def _nested_lookup(raw: dict[str, object], key: str) -> object | None:
            if key in raw:
                return raw[key]
            if "/" not in key:
                return None
            cur: object = raw
            for part in key.split("/"):
                if not isinstance(cur, dict) or part not in cur:
                    return None
                cur = cur[part]
            return cur

        def _coerce_float(value: object) -> float | None:
            if value is None:
                return None
            try:
                val = float(value)
            except Exception:
                return None
            if math.isnan(val):
                return None
            return val

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


DEFAULT_Y_KEY = "eval/episode_return"
AUTO_X_KEYS = ["_step"]
PROJECT_SUFFIXES = ["_FR_16_01", "_FR_19_01", "_FR_WPO_07_05", "_FR_REPPO_05_05", "_FR_PPO_27_04", "_FR_DPPO_29_04", "_FR_ME-WPO", "_FR_WPO"]
# Optional runtime-only suffix filter.
# Set to None (or []) to include all suffixes in runtime plots.
# Example:
# RUNTIME_PROJECT_SUFFIXES = ["_FR_REPPO_05_05", "_FR_PPO_27_04"]
RUNTIME_PROJECT_SUFFIXES = [ "_FR_REPPO_05_05", "_FR_16_01"]
# old DME PPO _FR_test_PPO --> new PPO _FR_PPO_14_04  --> very new PPO _FR_PPO_27_04
# old DME REPPO _FR_30_01 --> new _FR_REPPO_14_04
# _FR_16_01 REPPO-DIME
# _FR_19_01 REPPO
# _FR_24_01 DA-MDP: WPO 
# olde DME-REPPO _FR_REPPO_20_04 # latest run _FR_REPPO_05_05
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
        "reppo-dime-debug-1-<env_name>": {"alias": "REPPO-DIME", "color": "#ff0000"},
    },
    "_FR_19_01": {
        "reppo-<env_name>-reparam": {"alias": "REPPO", "color": "#8b1a1a"},
    },
    "_FR_WPO_07_05": {
        "reppo-dmerl-debug-1-<env_name>-WPO": {
            "alias": "DA-MDP: WPO (ours)",
            "color": "#1b7f3a",
        },
    },
    # "_FR_30_01": {
    #     "reppo-dmerl-debug-1-<env_name>-reparam": {
    #         "alias": "DA-MDP: REPPO (ours)",
    #         "color": "#8b1a1a",
    #     },
    # },
    "_FR_REPPO_20_04": {
        "reppo-dmerl-debug-1-<env_name>-reparam": {
            "alias": "DA-MDP: REPPO (ours)",
            "color": "#8b1a1a",
        },
    },
        "_FR_REPPO_05_05": {
        "reppo-dmerl-debug-1-<env_name>-reparam": {
            "alias": "DA-MDP: REPPO (ours)",
            "color": "#8b1a1a",
        },
    },
    "_FR_test_PPO": {
        "ppo-diff_ppo-<env_name>": {"alias": "DA-MDP: PPO (ours)", "color": "#3b528b"},
    },
    "_FR_PPO_27_04": {
        "ppo-diff_ppo-<env_name>": {"alias": "DA-MDP: PPO (ours)", "color": "#3b528b"},
    },
    "_FR_DPPO_29_04": {
        "ppo-diff_ppo-<env_name>": {"alias": "DPPO", "color": "#4c78a8"},
    },
    "_FR_ME-WPO": {
        "reppo-<env_name>-WPO": {"alias": "ME-WPO (ours)", "color": "#1b7f3a"},
    },
    "_FR_WPO": {
        "reppo-<env_name>-WPO": {"alias": "WPO", "color": "#9bd65a"},
    },
}
PPO_BRAX_LABEL = "PPO (r)"
FPO_LABEL = "FPO"
CSV_RESULTS_DIR = Path("results")
FPO_RESULTS_CSV = CSV_RESULTS_DIR / "FPO" / "fpo_full_results.csv"
FPO_REQUIRED_COLUMNS = {"Environment", "Seed", "Step", "Reward"}
FPO_MAX_INCLUDED_STEP = 50.0
RUNTIME_EXCLUDED_METHODS = {FPO_LABEL}
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
ALL_ENV_LEGEND_ROWS = 2
ALL_ENV_LEGEND_FONTSIZE = 12
LEGEND_HANDLELENGTH = 4.0
TITLE_FONTSIZE = 22
GRID_ALPHA = 0.9
alpha = 0.1
alpha_line = 0.7
GRID_LINESTYLE = ":"
GRID_LINEWIDTH = 2.0
GRID_COLS = 3
GRID_COLS_ALTERNATE = 4
METHOD_COLOR_OVERRIDES = {
    PPO_BRAX_LABEL: "#414487",
    FPO_LABEL: "#E60000",
}
METHOD_STYLE_OVERRIDES = {
    PPO_BRAX_LABEL: "--",
    FPO_LABEL: ":",
    "DPPO": "--",
    "DA-MDP: PPO (ours)": "-",
    "ME-WPO (ours)": "--",
    "DA-MDP: WPO (ours)": "-",
    "REPPO": "--",
    "DA-MDP: REPPO (ours)": "-",
    "REPPO-DIME": "-",
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
    "da-mdp: ppo": {"ppo (r)", "ppo_brax", "ppo"},
    "da-mdp: reppo": {"reppo"},
    "da-mdp: wpo": {"wpo", "me-wpo"},
}
REVIEWER_EXTRA_PROPOSED = {"reppo-dime"}
REVIEWER_SPECIAL_LINESTYLE = (0, (7, 2.2, 1.8, 2.2))
REVIEWER_METHOD_STYLE_OVERRIDES = {
    "wpo": REVIEWER_SPECIAL_LINESTYLE,
    "reppo-dime": REVIEWER_SPECIAL_LINESTYLE,
    "fpo": ":",
}
REVIEWER_METHOD_COLOR_OVERRIDES = {
    "fpo": "#E60000",
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
DPPO_DME_COMPARISON_DIR = "PPO_DPPO_comparison"
RUNTIME_COMPARISON_DIR = "runtime_comparison"
DPPO_ONLY_METHODS = {"DPPO", "DA-MDP: PPO (ours)"}
RUNTIME_TIME_UNIT_CHOICES = ("seconds", "minutes", "hours")
AUX_LOSS_FILTER_BY_ENV_AND_SUFFIX = {
    ("AcrobotSwingupSparse", "_FR_REPPO_20_04"): 0.1,
}
_AUX_LOSS_MULT_REGEX = re.compile(
    r"(?:^|[\s,])(?:hyperparameters\.)?aux_loss_mult\s*=\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


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
    parser.add_argument(
        "--runtime-key",
        default="auto",
        help=(
            "Metric key for runtime in seconds. Use 'auto' to try common keys "
            f"({', '.join(RUNTIME_KEYS)})."
        ),
    )
    parser.add_argument(
        "--runtime-time-unit",
        choices=RUNTIME_TIME_UNIT_CHOICES,
        default="hours",
        help="Time unit for runtime-x plots.",
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


def normalize_suffix_filter(suffixes: list[str] | None) -> set[str] | None:
    if suffixes is None:
        return None
    normalized = {suffix.strip() for suffix in suffixes if suffix and suffix.strip()}
    return normalized or None


def _to_float_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _collect_aux_loss_mult_values(obj: object) -> list[float]:
    values: list[float] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) == "aux_loss_mult":
                parsed = _to_float_or_none(value)
                if parsed is not None:
                    values.append(parsed)
            values.extend(_collect_aux_loss_mult_values(value))
        return values
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            values.extend(_collect_aux_loss_mult_values(item))
        return values
    if isinstance(obj, str):
        for match in _AUX_LOSS_MULT_REGEX.findall(obj):
            parsed = _to_float_or_none(match)
            if parsed is not None:
                values.append(parsed)
    return values


def run_matches_aux_loss_mult(
    run: wandb.apis.public.Run, target: float, tol: float = 1e-9
) -> bool:
    candidates = _collect_aux_loss_mult_values(getattr(run, "config", {}) or {})
    json_config = getattr(run, "json_config", None)
    if isinstance(json_config, str) and json_config:
        candidates.extend(_collect_aux_loss_mult_values(json_config))
    return any(abs(value - target) <= tol for value in candidates)


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
        or canonical_by_method[method].startswith("da-mdp:")
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
        if canonical in REVIEWER_METHOD_COLOR_OVERRIDES:
            colors[method] = REVIEWER_METHOD_COLOR_OVERRIDES[canonical]

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


def style_all_env_plot(
    ax: Axes,
    x_label: str,
    y_label: str,
    title: str,
    grid_alpha: float,
    grid_linewidth: float,
) -> None:
    fig = ax.figure
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, linestyle=GRID_LINESTYLE, alpha=grid_alpha, linewidth=grid_linewidth)

    top_rect = 0.92
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend_rows_target = 1 if len(handles) <= 2 else ALL_ENV_LEGEND_ROWS
        legend_cols = max(1, math.ceil(len(handles) / legend_rows_target))
        legend_rows = max(1, math.ceil(len(handles) / legend_cols))
        legend_fontsize = 9 #ALL_ENV_LEGEND_FONTSIZE
        top_rect = max(0.78, 0.90 - 0.05 * (legend_rows - 1))
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncols=legend_cols,
            fontsize=legend_fontsize,
            handlelength=LEGEND_HANDLELENGTH,
        )
        make_legend_clearer(legend)
    fig.tight_layout(rect=[0, 0, 1, top_rect])


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


def _load_fpo_seed_records(env_name: str) -> pd.DataFrame | None:
    if not FPO_RESULTS_CSV.is_file():
        print(f"Warning: FPO CSV not found: {FPO_RESULTS_CSV}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(FPO_RESULTS_CSV)
    except Exception as exc:
        print(f"Warning: failed to read FPO CSV {FPO_RESULTS_CSV} ({exc}).", file=sys.stderr)
        return None
    missing = sorted(FPO_REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        print(
            f"Warning: FPO CSV missing required columns ({', '.join(missing)}): "
            f"{FPO_RESULTS_CSV}",
            file=sys.stderr,
        )
        return None

    env_df = df[df["Environment"] == env_name].copy()
    if env_df.empty:
        return None
    env_df["Step"] = pd.to_numeric(env_df["Step"], errors="coerce")
    env_df["Reward"] = pd.to_numeric(env_df["Reward"], errors="coerce")
    env_df["Seed"] = pd.to_numeric(env_df["Seed"], errors="coerce")
    env_df = env_df.dropna(subset=["Step", "Reward", "Seed"])

    target_step = float(FPO_MAX_INCLUDED_STEP)
    synthetic_rows: list[dict[str, float]] = []
    grouped_seed_frames: list[pd.DataFrame] = []
    for seed, seed_df in env_df.groupby("Seed", sort=False):
        collapsed = (
            seed_df.groupby("Step", as_index=False)["Reward"].mean().sort_values("Step")
        )
        if collapsed.empty:
            continue
        grouped_seed_frames.append(
            pd.DataFrame(
                {
                    "Environment": env_name,
                    "Seed": float(seed),
                    "Step": collapsed["Step"],
                    "Reward": collapsed["Reward"],
                }
            )
        )
        if (collapsed["Step"] == target_step).any():
            continue

        lower = collapsed[collapsed["Step"] < target_step]
        upper = collapsed[collapsed["Step"] > target_step]
        if lower.empty or upper.empty:
            continue
        lo = lower.iloc[-1]
        hi = upper.iloc[0]
        lo_step = float(lo["Step"])
        hi_step = float(hi["Step"])
        if hi_step <= lo_step:
            continue
        lo_reward = float(lo["Reward"])
        hi_reward = float(hi["Reward"])
        interp_reward = lo_reward + (target_step - lo_step) * (hi_reward - lo_reward) / (
            hi_step - lo_step
        )
        synthetic_rows.append(
            {
                "Environment": env_name,
                "Seed": float(seed),
                "Step": target_step,
                "Reward": interp_reward,
            }
        )

    if grouped_seed_frames:
        env_df = pd.concat(grouped_seed_frames, ignore_index=True)
    if synthetic_rows:
        env_df = pd.concat([env_df, pd.DataFrame(synthetic_rows)], ignore_index=True)

    env_df = env_df[env_df["Step"] <= target_step]
    if env_df.empty:
        return None

    env_df["step"] = env_df["Step"] / target_step * float(PLOT_MAX_STEPS)
    env_df["value"] = env_df["Reward"]
    env_df["method"] = FPO_LABEL
    env_df["run_id"] = env_df["Seed"].astype(int).map(
        lambda seed: f"fpo:{env_name}:seed{seed}"
    )
    return env_df[["step", "value", "method", "run_id"]]


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


def _summary_dict(run: wandb.apis.public.Run) -> dict[str, object]:
    try:
        return dict(run.summary or {})
    except Exception:
        return {}


def seconds_to_unit(seconds: float, unit: str) -> float:
    if unit == "seconds":
        return seconds
    if unit == "minutes":
        return seconds / 60.0
    return seconds / 3600.0


def runtime_axis_label(unit: str) -> str:
    if unit == "seconds":
        return "runtime (s)"
    if unit == "minutes":
        return "runtime (min)"
    return "runtime (h)"


def filter_positive_x(data: pd.DataFrame, x_column: str) -> pd.DataFrame:
    if data.empty:
        return data
    return data[data[x_column] > 0].copy()


def min_positive_x(data: pd.DataFrame, x_column: str) -> float | None:
    if data.empty or x_column not in data.columns:
        return None
    positive = data[data[x_column] > 0][x_column]
    if positive.empty:
        return None
    return float(positive.min())


def filter_methods_frame(data: pd.DataFrame, allowed_methods: set[str]) -> pd.DataFrame:
    if data.empty:
        return data
    return data[data["method"].isin(allowed_methods)].copy()


def exclude_methods_frame(data: pd.DataFrame, excluded_methods: set[str]) -> pd.DataFrame:
    if data.empty:
        return data
    return data[~data["method"].isin(excluded_methods)].copy()


def filter_records_by_project_suffix(
    data: pd.DataFrame, allowed_suffixes: set[str] | None
) -> pd.DataFrame:
    if data.empty or allowed_suffixes is None:
        return data
    if "project_suffix" not in data.columns:
        return data.iloc[0:0].copy()
    return data[data["project_suffix"].isin(allowed_suffixes)].copy()


def build_method_run_counts_from_records(data: pd.DataFrame) -> dict[str, set[str]]:
    method_run_counts: dict[str, set[str]] = defaultdict(set)
    if data.empty or "method" not in data.columns or "run_id" not in data.columns:
        return method_run_counts
    unique_pairs = data[["method", "run_id"]].drop_duplicates()
    for row in unique_pairs.itertuples(index=False):
        method_run_counts[str(row.method)].add(str(row.run_id))
    return method_run_counts


def _plot_method_curves(
    ax: Axes,
    merged: pd.DataFrame,
    method_order: list[str],
    method_run_counts: dict[str, set[str]],
    color_by_method: dict[str, str],
    style_by_method: dict[str, object],
    x_column: str = "step",
    alpha_fill: float = alpha,
    alpha_series: float = alpha_line,
    use_run_counts_in_label: bool = True,
) -> dict[str, Line2D]:
    legend_handles: dict[str, Line2D] = {}
    for method in ordered_methods_present(merged, method_order):
        method_df = merged[merged["method"] == method].sort_values(x_column)
        if method_df.empty:
            continue
        run_count = len(method_run_counts.get(method, set()))
        label = f"{method} (n={run_count})" if use_run_counts_in_label else method
        color = color_by_method.get(method)
        linestyle = style_by_method.get(method, "-")
        (line,) = ax.plot(
            method_df[x_column],
            method_df["value"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=LINE_WIDTH,
            alpha=alpha_series,
        )
        legend_handles[method] = line
        if method_df["stderr"].notna().any():
            ax.fill_between(
                method_df[x_column],
                method_df["value"] - method_df["stderr"],
                method_df["value"] + method_df["stderr"],
                color=color,
                alpha=alpha_fill,
            )
    return legend_handles


def _save_multi_env_grid_plot(
    env_results: list[dict[str, object]],
    method_order: list[str],
    color_by_method: dict[str, str],
    style_by_method: dict[str, object],
    method_run_counts_key: str,
    merged_key: str,
    x_column: str,
    x_label: str,
    y_label: str,
    output_path: str,
    ncols: int,
    grid_alpha: float,
    grid_linewidth: float,
    xlim_left: float = 0.0,
    xlim_right: float | None = None,
    empty_text: str | None = None,
    x_scale: str = "linear",
    require_positive_x: bool = False,
) -> None:
    num_envs = len(env_results)
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
    legend_handles: dict[str, Line2D] = {}

    for idx, result in enumerate(env_results):
        ax = axes_list[idx]
        env_name = str(result["env_name"])
        merged = result.get(merged_key, pd.DataFrame())
        if not isinstance(merged, pd.DataFrame):
            merged = pd.DataFrame()
        if require_positive_x:
            merged = filter_positive_x(merged, x_column)
        method_run_counts = result.get(method_run_counts_key, {})
        if merged.empty and empty_text is not None:
            ax.set_title(env_name, fontsize=TITLE_FONTSIZE)
            ax.text(0.5, 0.5, empty_text, ha="center", va="center", transform=ax.transAxes)
            ax.grid(True, linestyle=GRID_LINESTYLE, alpha=grid_alpha, linewidth=grid_linewidth)
            if x_scale == "log":
                ax.set_xscale("log")
            if xlim_left is not None or xlim_right is not None:
                ax.set_xlim(left=xlim_left, right=xlim_right)
            continue
        handles = _plot_method_curves(
            ax,
            merged,
            method_order,
            method_run_counts,
            color_by_method,
            style_by_method,
            x_column=x_column,
            use_run_counts_in_label=False,
        )
        for method, line in handles.items():
            if method not in legend_handles:
                legend_handles[method] = line

        ax.set_title(env_name, fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, linestyle=GRID_LINESTYLE, alpha=grid_alpha, linewidth=grid_linewidth)
        if x_scale == "log":
            ax.set_xscale("log")
        subplot_xlim_left = xlim_left
        if subplot_xlim_left is None and x_scale == "log":
            subplot_xlim_left = min_positive_x(merged, x_column)
        if subplot_xlim_left is not None or xlim_right is not None:
            ax.set_xlim(left=subplot_xlim_left, right=xlim_right)

    for idx in range(num_envs, len(axes_list)):
        fig.delaxes(axes_list[idx])

    fig.supxlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE)
    fig.supylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    top_rect = 0.92
    if legend_handles:
        handles = [legend_handles[m] for m in method_order if m in legend_handles]
        labels = [h.get_label() for h in handles]
        legend_cols = max(1, math.ceil(len(handles) / 2))
        legend_rows = max(1, math.ceil(len(handles) / legend_cols))
        top_rect = max(0.78, 0.90 - 0.05 * (legend_rows - 1))
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=legend_cols,
            fontsize=LEGEND_FONTSIZE,
            handlelength=LEGEND_HANDLELENGTH,
        )
        make_legend_clearer(legend)
    fig.tight_layout(rect=[0, 0, 1, top_rect])
    fig.savefig(output_path, dpi=800, bbox_inches="tight")
    print(f"Saved grid plot to {output_path}")
    plt.close(fig)


def add_runtime_x(
    merged: pd.DataFrame,
    runtime_mean_by_env_method: dict[tuple[str, str], float],
    env_name: str,
    unit: str,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing: list[str] = []
    for method in merged["method"].drop_duplicates().tolist():
        runtime_s = runtime_mean_by_env_method.get((env_name, method))
        if runtime_s is None or runtime_s <= 0:
            missing.append(method)
            continue
        runtime_u = seconds_to_unit(runtime_s, unit)
        method_df = merged[merged["method"] == method].copy()
        method_df["runtime_x"] = method_df["step"] / float(PLOT_MAX_STEPS) * runtime_u
        rows.append(method_df)
    if missing and verbose:
        missing_str = ", ".join(sorted(missing))
        print(
            f"Warning: missing runtime for env={env_name}, methods={missing_str}; "
            "skipping them in runtime-x plots.",
            file=sys.stderr,
        )
    if not rows:
        return pd.DataFrame(columns=[*merged.columns, "runtime_x"]), missing
    return pd.concat(rows, ignore_index=True), missing


def add_runtime_x_overall(
    merged: pd.DataFrame,
    runtime_mean_by_method_s: dict[str, float],
    unit: str,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing: list[str] = []
    for method in merged["method"].drop_duplicates().tolist():
        runtime_s = runtime_mean_by_method_s.get(method)
        if runtime_s is None or runtime_s <= 0:
            missing.append(method)
            continue
        runtime_u = seconds_to_unit(runtime_s, unit)
        method_df = merged[merged["method"] == method].copy()
        method_df["runtime_x"] = method_df["step"] / float(PLOT_MAX_STEPS) * runtime_u
        rows.append(method_df)
    if missing and verbose:
        missing_str = ", ".join(sorted(missing))
        print(
            f"Warning: missing average runtime for methods={missing_str}; "
            "skipping them in overall runtime-x plots.",
            file=sys.stderr,
        )
    if not rows:
        return pd.DataFrame(columns=[*merged.columns, "runtime_x"]), missing
    return pd.concat(rows, ignore_index=True), missing


def make_filename(base: str, out_stem: str | None, out_ext: str) -> str:
    if out_stem is None:
        return base
    ext = out_ext if out_ext else ".png"
    return f"{out_stem}_{base.rsplit('.', 1)[0]}{ext}"


def main() -> int:
    args = parse_args()
    api = wandb.Api()
    runs_by_suffix = normalize_runs_by_suffix(RUNS_BY_SUFFIX)
    runtime_project_suffixes = normalize_suffix_filter(RUNTIME_PROJECT_SUFFIXES)
    if runtime_project_suffixes is not None:
        selected_suffixes = ", ".join(sorted(runtime_project_suffixes))
        print(f"Runtime plots: filtering to project suffixes: {selected_suffixes}")
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
    dppo_mean_figures_dir = os.path.join(
        figures_dir, DPPO_DME_COMPARISON_DIR, f"{MEAN_FIGURES_SUBDIR}{style_dir_suffix}"
    )
    dppo_iqm_figures_dir = os.path.join(
        figures_dir, DPPO_DME_COMPARISON_DIR, f"{IQM_FIGURES_SUBDIR}{style_dir_suffix}"
    )
    runtime_mean_figures_dir = os.path.join(
        figures_dir, RUNTIME_COMPARISON_DIR, f"{MEAN_FIGURES_SUBDIR}{style_dir_suffix}"
    )
    runtime_iqm_figures_dir = os.path.join(
        figures_dir, RUNTIME_COMPARISON_DIR, f"{IQM_FIGURES_SUBDIR}{style_dir_suffix}"
    )
    os.makedirs(mean_figures_dir, exist_ok=True)
    os.makedirs(iqm_figures_dir, exist_ok=True)
    os.makedirs(dppo_mean_figures_dir, exist_ok=True)
    os.makedirs(dppo_iqm_figures_dir, exist_ok=True)
    os.makedirs(runtime_mean_figures_dir, exist_ok=True)
    os.makedirs(runtime_iqm_figures_dir, exist_ok=True)
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
    runtime_records: list[dict[str, object]] = []

    for env_name, projects in env_projects.items():
        records = []
        method_run_counts = defaultdict(set)
        skipped = 0
        for project in projects:
            project_path = resolve_project_path(api, args.entity, project)
            run_name_counts = defaultdict(int)
            project_suffix = resolve_project_suffix(project)
            aux_loss_filter_target = AUX_LOSS_FILTER_BY_ENV_AND_SUFFIX.get(
                (env_name, project_suffix or "")
            )
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
                if (
                    aux_loss_filter_target is not None
                    and not run_matches_aux_loss_mult(run, aux_loss_filter_target)
                ):
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
                df["project_suffix"] = project_suffix or ""
                records.append(df)
                runtime_s = extract_runtime_seconds(_summary_dict(run), args.runtime_key)
                if runtime_s is not None:
                    runtime_records.append(
                        {
                            "env_name": env_name,
                            "method": method_name,
                            "run_id": run.id,
                            "project_suffix": project_suffix or "",
                            "runtime_s": runtime_s,
                        }
                    )
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
                csv_df["project_suffix"] = "csv"
                records.append(csv_df)
                method_run_counts[PPO_BRAX_LABEL].add(csv_df["run_id"].iloc[0])
            all_methods.add(PPO_BRAX_LABEL)

        fpo_df = _load_fpo_seed_records(env_name)
        if fpo_df is not None and not fpo_df.empty:
            fpo_df = fpo_df.copy()
            fpo_df["project_suffix"] = "fpo"
            records.append(fpo_df)
            for run_id in fpo_df["run_id"].drop_duplicates():
                method_run_counts[FPO_LABEL].add(str(run_id))
            all_methods.add(FPO_LABEL)

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

    runtime_mean_by_env_method: dict[tuple[str, str], float] = {}
    runtime_count_by_env_method: dict[tuple[str, str], int] = {}
    runtime_mean_by_method_s: dict[str, float] = {}
    if runtime_records:
        runtime_df = pd.DataFrame.from_records(runtime_records)
        runtime_df = filter_records_by_project_suffix(runtime_df, runtime_project_suffixes)
        grouped_runtime = runtime_df.groupby(["env_name", "method"])["runtime_s"]
        runtime_mean_by_env_method = grouped_runtime.mean().to_dict()
        runtime_count_by_env_method = grouped_runtime.count().to_dict()
        method_means: dict[str, list[float]] = defaultdict(list)
        for (env_name, method), mean_runtime in runtime_mean_by_env_method.items():
            del env_name
            method_means[method].append(float(mean_runtime))
        runtime_mean_by_method_s = {
            method: float(pd.Series(values).mean()) for method, values in method_means.items()
        }

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
        runtime_data = filter_records_by_project_suffix(data, runtime_project_suffixes)
        runtime_method_run_counts = build_method_run_counts_from_records(runtime_data)
        merged_runtime_source = aggregate_method_step(runtime_data, use_iqm=False)
        merged_iqm_runtime_source = aggregate_method_step(runtime_data, use_iqm=True)
        merged_runtime_source = exclude_methods_frame(merged_runtime_source, RUNTIME_EXCLUDED_METHODS)
        merged_iqm_runtime_source = exclude_methods_frame(merged_iqm_runtime_source, RUNTIME_EXCLUDED_METHODS)

        result["merged"] = merged
        result["merged_iqm"] = merged_iqm
        result["runtime_method_run_counts"] = runtime_method_run_counts
        merged_runtime, _ = add_runtime_x(
            merged_runtime_source,
            runtime_mean_by_env_method,
            env_name,
            args.runtime_time_unit,
            verbose=True,
        )
        merged_iqm_runtime, _ = add_runtime_x(
            merged_iqm_runtime_source,
            runtime_mean_by_env_method,
            env_name,
            args.runtime_time_unit,
            verbose=True,
        )
        result["merged_runtime"] = merged_runtime
        result["merged_iqm_runtime"] = merged_iqm_runtime

        plt.figure(figsize=(9, 5))
        _plot_method_curves(
            plt.gca(),
            merged,
            method_order,
            method_run_counts,
            color_by_method,
            style_by_method,
            x_column="step",
        )

        plt.xlabel("env calls", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("episode return", fontsize=AXIS_LABEL_FONTSIZE)
        plt.title(f"{env_name}", fontsize=TITLE_FONTSIZE)
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
        _plot_method_curves(
            plt.gca(),
            merged_iqm,
            method_order,
            method_run_counts,
            color_by_method,
            style_by_method,
            x_column="step",
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

        dppo_merged = filter_methods_frame(merged, DPPO_ONLY_METHODS)
        dppo_merged_iqm = filter_methods_frame(merged_iqm, DPPO_ONLY_METHODS)
        dppo_method_order = [m for m in method_order if m in DPPO_ONLY_METHODS]
        if not dppo_merged.empty:
            plt.figure(figsize=(9, 5))
            _plot_method_curves(
                plt.gca(),
                dppo_merged,
                dppo_method_order,
                method_run_counts,
                color_by_method,
                style_by_method,
                x_column="step",
            )
            plt.xlabel("env calls", fontsize=AXIS_LABEL_FONTSIZE)
            plt.ylabel("episode return", fontsize=AXIS_LABEL_FONTSIZE)
            plt.title(env_name, fontsize=TITLE_FONTSIZE)
            legend = plt.legend(
                loc="lower right",
                ncol=1,
                fontsize=LEGEND_FONTSIZE,
                handlelength=LEGEND_HANDLELENGTH,
            )
            make_legend_clearer(legend)
            plt.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            plt.grid(True, linestyle=GRID_LINESTYLE, alpha=grid_alpha, linewidth=grid_linewidth)
            plt.xlim(left=0, right=PLOT_MAX_STEPS)
            plt.tight_layout()
            dppo_mean_name = f"dime_{env_name}_DPPO_DMEPPO_avg_eval_return.png"
            dppo_mean_output = os.path.join(dppo_mean_figures_dir, dppo_mean_name)
            plt.savefig(dppo_mean_output, dpi=800)
            print(f"Saved DPPO-only plot to {dppo_mean_output}")
            plt.close()

        if not dppo_merged_iqm.empty:
            plt.figure(figsize=(9, 5))
            _plot_method_curves(
                plt.gca(),
                dppo_merged_iqm,
                dppo_method_order,
                method_run_counts,
                color_by_method,
                style_by_method,
                x_column="step",
            )
            plt.xlabel("env calls", fontsize=AXIS_LABEL_FONTSIZE)
            plt.ylabel("episode return (IQM)", fontsize=AXIS_LABEL_FONTSIZE)
            plt.title(env_name, fontsize=TITLE_FONTSIZE)
            legend = plt.legend(
                loc="lower right",
                ncol=1,
                fontsize=LEGEND_FONTSIZE,
                handlelength=LEGEND_HANDLELENGTH,
            )
            make_legend_clearer(legend)
            plt.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            plt.grid(True, linestyle=GRID_LINESTYLE, alpha=grid_alpha, linewidth=grid_linewidth)
            plt.xlim(left=0, right=PLOT_MAX_STEPS)
            plt.tight_layout()
            dppo_iqm_name = f"dime_{env_name}_DPPO_DMEPPO_iqm_eval_return.png"
            dppo_iqm_output = os.path.join(dppo_iqm_figures_dir, dppo_iqm_name)
            plt.savefig(dppo_iqm_output, dpi=800)
            print(f"Saved DPPO-only IQM plot to {dppo_iqm_output}")
            plt.close()

        if not merged_runtime.empty:
            plt.figure(figsize=(9, 5))
            _plot_method_curves(
                plt.gca(),
                merged_runtime,
                method_order,
                runtime_method_run_counts,
                color_by_method,
                style_by_method,
                x_column="runtime_x",
            )
            plt.xlabel(runtime_axis_label(args.runtime_time_unit), fontsize=AXIS_LABEL_FONTSIZE)
            plt.ylabel("episode return", fontsize=AXIS_LABEL_FONTSIZE)
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
            plt.xlim(left=0)
            plt.tight_layout()
            runtime_mean_name = f"dime_{env_name}_methods_avg_eval_return_runtime.png"
            runtime_mean_output = os.path.join(runtime_mean_figures_dir, runtime_mean_name)
            plt.savefig(runtime_mean_output, dpi=800)
            print(f"Saved runtime-x plot to {runtime_mean_output}")
            plt.close()

            merged_runtime_log = filter_positive_x(merged_runtime, "runtime_x")
            if not merged_runtime_log.empty:
                plt.figure(figsize=(9, 5))
                _plot_method_curves(
                    plt.gca(),
                    merged_runtime_log,
                    method_order,
                    runtime_method_run_counts,
                    color_by_method,
                    style_by_method,
                    x_column="runtime_x",
                )
                plt.xlabel(runtime_axis_label(args.runtime_time_unit), fontsize=AXIS_LABEL_FONTSIZE)
                plt.ylabel("episode return", fontsize=AXIS_LABEL_FONTSIZE)
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
                plt.xscale("log")
                left = min_positive_x(merged_runtime_log, "runtime_x")
                if left is not None:
                    plt.xlim(left=left)
                plt.tight_layout()
                runtime_mean_log_name = (
                    f"dime_{env_name}_methods_avg_eval_return_runtime_logx.png"
                )
                runtime_mean_log_output = os.path.join(
                    runtime_mean_figures_dir, runtime_mean_log_name
                )
                plt.savefig(runtime_mean_log_output, dpi=800)
                print(f"Saved runtime-x log plot to {runtime_mean_log_output}")
                plt.close()

        if not merged_iqm_runtime.empty:
            plt.figure(figsize=(9, 5))
            _plot_method_curves(
                plt.gca(),
                merged_iqm_runtime,
                method_order,
                runtime_method_run_counts,
                color_by_method,
                style_by_method,
                x_column="runtime_x",
            )
            plt.xlabel(runtime_axis_label(args.runtime_time_unit), fontsize=AXIS_LABEL_FONTSIZE)
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
            plt.xlim(left=0)
            plt.tight_layout()
            runtime_iqm_name = f"dime_{env_name}_methods_iqm_eval_return_runtime.png"
            runtime_iqm_output = os.path.join(runtime_iqm_figures_dir, runtime_iqm_name)
            plt.savefig(runtime_iqm_output, dpi=800)
            print(f"Saved runtime-x IQM plot to {runtime_iqm_output}")
            plt.close()

            merged_iqm_runtime_log = filter_positive_x(merged_iqm_runtime, "runtime_x")
            if not merged_iqm_runtime_log.empty:
                plt.figure(figsize=(9, 5))
                _plot_method_curves(
                    plt.gca(),
                    merged_iqm_runtime_log,
                    method_order,
                    runtime_method_run_counts,
                    color_by_method,
                    style_by_method,
                    x_column="runtime_x",
                )
                plt.xlabel(runtime_axis_label(args.runtime_time_unit), fontsize=AXIS_LABEL_FONTSIZE)
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
                plt.xscale("log")
                left = min_positive_x(merged_iqm_runtime_log, "runtime_x")
                if left is not None:
                    plt.xlim(left=left)
                plt.tight_layout()
                runtime_iqm_log_name = (
                    f"dime_{env_name}_methods_iqm_eval_return_runtime_logx.png"
                )
                runtime_iqm_log_output = os.path.join(
                    runtime_iqm_figures_dir, runtime_iqm_log_name
                )
                plt.savefig(runtime_iqm_log_output, dpi=800)
                print(f"Saved runtime-x IQM log plot to {runtime_iqm_log_output}")
                plt.close()

        if skipped:
            print(f"Skipped {skipped} runs without usable data in {env_name}.")

    if multi_env and env_results:
        num_envs = len(env_results)
        default_cols = min(GRID_COLS, num_envs)
        layout_cols = [default_cols]
        alt_cols = min(GRID_COLS_ALTERNATE, num_envs)
        if alt_cols not in layout_cols:
            layout_cols.append(alt_cols)

        for ncols in layout_cols:
            layout_suffix = "" if ncols == default_cols else f"_{ncols}cols"
            if out_stem is not None:
                grid_output = os.path.join(
                    mean_figures_dir, f"{out_stem}_grid{layout_suffix}{out_ext}"
                )
                grid_iqm_output = os.path.join(
                    iqm_figures_dir, f"{out_stem}_grid_iqm{layout_suffix}{out_ext}"
                )
            else:
                grid_output = os.path.join(
                    mean_figures_dir, f"all_envs_methods_grid_eval_return{layout_suffix}.png"
                )
                grid_iqm_output = os.path.join(
                    iqm_figures_dir,
                    f"all_envs_methods_grid_iqm_eval_return{layout_suffix}.png",
                )

            runtime_grid_output = os.path.join(
                runtime_mean_figures_dir,
                f"all_envs_methods_grid_eval_return_runtime{layout_suffix}.png",
            )
            runtime_grid_log_output = os.path.join(
                runtime_mean_figures_dir,
                f"all_envs_methods_grid_eval_return_runtime_logx{layout_suffix}.png",
            )
            runtime_grid_iqm_output = os.path.join(
                runtime_iqm_figures_dir,
                f"all_envs_methods_grid_iqm_eval_return_runtime{layout_suffix}.png",
            )
            runtime_grid_iqm_log_output = os.path.join(
                runtime_iqm_figures_dir,
                f"all_envs_methods_grid_iqm_eval_return_runtime_logx{layout_suffix}.png",
            )

            _save_multi_env_grid_plot(
                env_results=env_results,
                method_order=method_order,
                color_by_method=color_by_method,
                style_by_method=style_by_method,
                method_run_counts_key="method_run_counts",
                merged_key="merged",
                x_column="step",
                x_label="env calls",
                y_label="episode return",
                output_path=grid_output,
                ncols=ncols,
                grid_alpha=grid_alpha,
                grid_linewidth=grid_linewidth,
                xlim_left=0,
                xlim_right=PLOT_MAX_STEPS,
            )
            _save_multi_env_grid_plot(
                env_results=env_results,
                method_order=method_order,
                color_by_method=color_by_method,
                style_by_method=style_by_method,
                method_run_counts_key="method_run_counts",
                merged_key="merged_iqm",
                x_column="step",
                x_label="env calls",
                y_label="episode return (IQM)",
                output_path=grid_iqm_output,
                ncols=ncols,
                grid_alpha=grid_alpha,
                grid_linewidth=grid_linewidth,
                xlim_left=0,
                xlim_right=PLOT_MAX_STEPS,
            )
            _save_multi_env_grid_plot(
                env_results=env_results,
                method_order=method_order,
                color_by_method=color_by_method,
                style_by_method=style_by_method,
                method_run_counts_key="runtime_method_run_counts",
                merged_key="merged_runtime",
                x_column="runtime_x",
                x_label=runtime_axis_label(args.runtime_time_unit),
                y_label="episode return",
                output_path=runtime_grid_output,
                ncols=ncols,
                grid_alpha=grid_alpha,
                grid_linewidth=grid_linewidth,
                xlim_left=0,
                xlim_right=None,
                empty_text="no runtime",
            )
            _save_multi_env_grid_plot(
                env_results=env_results,
                method_order=method_order,
                color_by_method=color_by_method,
                style_by_method=style_by_method,
                method_run_counts_key="runtime_method_run_counts",
                merged_key="merged_iqm_runtime",
                x_column="runtime_x",
                x_label=runtime_axis_label(args.runtime_time_unit),
                y_label="episode return (IQM)",
                output_path=runtime_grid_iqm_output,
                ncols=ncols,
                grid_alpha=grid_alpha,
                grid_linewidth=grid_linewidth,
                xlim_left=0,
                xlim_right=None,
                empty_text="no runtime",
            )
            _save_multi_env_grid_plot(
                env_results=env_results,
                method_order=method_order,
                color_by_method=color_by_method,
                style_by_method=style_by_method,
                method_run_counts_key="runtime_method_run_counts",
                merged_key="merged_runtime",
                x_column="runtime_x",
                x_label=runtime_axis_label(args.runtime_time_unit),
                y_label="episode return",
                output_path=runtime_grid_log_output,
                ncols=ncols,
                grid_alpha=grid_alpha,
                grid_linewidth=grid_linewidth,
                xlim_left=None,
                xlim_right=None,
                empty_text="no runtime",
                x_scale="log",
                require_positive_x=True,
            )
            _save_multi_env_grid_plot(
                env_results=env_results,
                method_order=method_order,
                color_by_method=color_by_method,
                style_by_method=style_by_method,
                method_run_counts_key="runtime_method_run_counts",
                merged_key="merged_iqm_runtime",
                x_column="runtime_x",
                x_label=runtime_axis_label(args.runtime_time_unit),
                y_label="episode return (IQM)",
                output_path=runtime_grid_iqm_log_output,
                ncols=ncols,
                grid_alpha=grid_alpha,
                grid_linewidth=grid_linewidth,
                xlim_left=None,
                xlim_right=None,
                empty_text="no runtime",
                x_scale="log",
                require_positive_x=True,
            )

    if multi_env and env_results:
        overall_records = pd.concat([r["records"] for r in env_results], ignore_index=True)
        overall_merged = aggregate_method_step(overall_records, use_iqm=False)
        overall_merged_iqm = aggregate_method_step(overall_records, use_iqm=True)
        overall_runtime_records = filter_records_by_project_suffix(
            overall_records, runtime_project_suffixes
        )
        overall_runtime_source = aggregate_method_step(overall_runtime_records, use_iqm=False)
        overall_iqm_runtime_source = aggregate_method_step(overall_runtime_records, use_iqm=True)
        overall_runtime_source = exclude_methods_frame(overall_runtime_source, RUNTIME_EXCLUDED_METHODS)
        overall_iqm_runtime_source = exclude_methods_frame(
            overall_iqm_runtime_source, RUNTIME_EXCLUDED_METHODS
        )
        overall_merged_runtime, _ = add_runtime_x_overall(
            overall_runtime_source,
            runtime_mean_by_method_s,
            args.runtime_time_unit,
            verbose=True,
        )
        overall_merged_iqm_runtime, _ = add_runtime_x_overall(
            overall_iqm_runtime_source,
            runtime_mean_by_method_s,
            args.runtime_time_unit,
            verbose=True,
        )

        overall_method_runs: dict[str, set[str]] = defaultdict(set)
        for result in env_results:
            for method, runs in result["method_run_counts"].items():
                overall_method_runs[method].update(runs)
        overall_run_counts = {method: len(runs) for method, runs in overall_method_runs.items()}
        overall_runtime_method_runs: dict[str, set[str]] = defaultdict(set)
        for result in env_results:
            runtime_method_runs = result.get("runtime_method_run_counts", {})
            for method, runs in runtime_method_runs.items():
                overall_runtime_method_runs[method].update(runs)

        plt.figure(figsize=(9, 5))
        _plot_method_curves(
            plt.gca(),
            overall_merged,
            method_order,
            overall_method_runs,
            color_by_method,
            style_by_method,
            x_column="step",
            use_run_counts_in_label=False,
        )

        style_all_env_plot(
            plt.gca(),
            x_label="env calls",
            y_label="episode return",
            title="All environments",
            grid_alpha=grid_alpha,
            grid_linewidth=grid_linewidth,
        )

        output_path = os.path.join(mean_figures_dir, "all_envs_methods_avg_eval_return.png")
        plt.savefig(output_path, dpi=800)
        print(f"Saved plot to {output_path}")
        plt.close()

        plt.figure(figsize=(9, 5))
        _plot_method_curves(
            plt.gca(),
            overall_merged_iqm,
            method_order,
            overall_method_runs,
            color_by_method,
            style_by_method,
            x_column="step",
            use_run_counts_in_label=False,
        )

        style_all_env_plot(
            plt.gca(),
            x_label="env calls",
            y_label="episode return (IQM)",
            title="All environments",
            grid_alpha=grid_alpha,
            grid_linewidth=grid_linewidth,
        )

        output_iqm_path = os.path.join(iqm_figures_dir, "all_envs_methods_iqm_eval_return.png")
        plt.savefig(output_iqm_path, dpi=800)
        print(f"Saved IQM plot to {output_iqm_path}")
        plt.close()

        if not overall_merged_runtime.empty:
            plt.figure(figsize=(9, 5))
            _plot_method_curves(
                plt.gca(),
                overall_merged_runtime,
                method_order,
                overall_runtime_method_runs,
                color_by_method,
                style_by_method,
                x_column="runtime_x",
                use_run_counts_in_label=False,
            )
            plt.xlim(left=0)
            style_all_env_plot(
                plt.gca(),
                x_label=runtime_axis_label(args.runtime_time_unit),
                y_label="episode return",
                title="All environments",
                grid_alpha=grid_alpha,
                grid_linewidth=grid_linewidth,
            )
            output_runtime_path = os.path.join(
                runtime_mean_figures_dir, "all_envs_methods_avg_eval_return_runtime.png"
            )
            plt.savefig(output_runtime_path, dpi=800)
            print(f"Saved runtime-x overall plot to {output_runtime_path}")
            plt.close()

            overall_merged_runtime_log = filter_positive_x(
                overall_merged_runtime, "runtime_x"
            )
            if not overall_merged_runtime_log.empty:
                plt.figure(figsize=(9, 5))
                _plot_method_curves(
                    plt.gca(),
                    overall_merged_runtime_log,
                    method_order,
                    overall_runtime_method_runs,
                    color_by_method,
                    style_by_method,
                    x_column="runtime_x",
                    use_run_counts_in_label=False,
                )
                plt.xscale("log")
                left = min_positive_x(overall_merged_runtime_log, "runtime_x")
                if left is not None:
                    plt.xlim(left=left)
                style_all_env_plot(
                    plt.gca(),
                    x_label=runtime_axis_label(args.runtime_time_unit),
                    y_label="episode return",
                    title="All environments",
                    grid_alpha=grid_alpha,
                    grid_linewidth=grid_linewidth,
                )
                output_runtime_log_path = os.path.join(
                    runtime_mean_figures_dir,
                    "all_envs_methods_avg_eval_return_runtime_logx.png",
                )
                plt.savefig(output_runtime_log_path, dpi=800)
                print(f"Saved runtime-x overall log plot to {output_runtime_log_path}")
                plt.close()

        if not overall_merged_iqm_runtime.empty:
            plt.figure(figsize=(9, 5))
            _plot_method_curves(
                plt.gca(),
                overall_merged_iqm_runtime,
                method_order,
                overall_runtime_method_runs,
                color_by_method,
                style_by_method,
                x_column="runtime_x",
                use_run_counts_in_label=False,
            )
            plt.xlim(left=0)
            style_all_env_plot(
                plt.gca(),
                x_label=runtime_axis_label(args.runtime_time_unit),
                y_label="episode return (IQM)",
                title="All environments",
                grid_alpha=grid_alpha,
                grid_linewidth=grid_linewidth,
            )
            output_runtime_iqm_path = os.path.join(
                runtime_iqm_figures_dir, "all_envs_methods_iqm_eval_return_runtime.png"
            )
            plt.savefig(output_runtime_iqm_path, dpi=800)
            print(f"Saved runtime-x overall IQM plot to {output_runtime_iqm_path}")
            plt.close()

            overall_merged_iqm_runtime_log = filter_positive_x(
                overall_merged_iqm_runtime, "runtime_x"
            )
            if not overall_merged_iqm_runtime_log.empty:
                plt.figure(figsize=(9, 5))
                _plot_method_curves(
                    plt.gca(),
                    overall_merged_iqm_runtime_log,
                    method_order,
                    overall_runtime_method_runs,
                    color_by_method,
                    style_by_method,
                    x_column="runtime_x",
                    use_run_counts_in_label=False,
                )
                plt.xscale("log")
                left = min_positive_x(overall_merged_iqm_runtime_log, "runtime_x")
                if left is not None:
                    plt.xlim(left=left)
                style_all_env_plot(
                    plt.gca(),
                    x_label=runtime_axis_label(args.runtime_time_unit),
                    y_label="episode return (IQM)",
                    title="All environments",
                    grid_alpha=grid_alpha,
                    grid_linewidth=grid_linewidth,
                )
                output_runtime_iqm_log_path = os.path.join(
                    runtime_iqm_figures_dir,
                    "all_envs_methods_iqm_eval_return_runtime_logx.png",
                )
                plt.savefig(output_runtime_iqm_log_path, dpi=800)
                print(f"Saved runtime-x overall IQM log plot to {output_runtime_iqm_log_path}")
                plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
