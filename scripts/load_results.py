from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd


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


def _normalize_token(s: str) -> str:
    return s.strip().lower()


def _iter_matching_csvs(
    results_dir: Path, env: str, method: str | None, spec: str | None
) -> Iterable[Path]:
    env_norm = _normalize_token(env)
    method_norm = _normalize_token(method) if method else None
    spec_norm = _normalize_token(spec) if spec else None

    for path in results_dir.glob("*.csv"):
        stem = path.stem
        stem_norm = stem.lower()
        if not stem_norm.startswith(env_norm):
            continue

        remainder = stem_norm[len(env_norm) :]
        tokens = [t for t in remainder.split("_") if t]
        if method_norm and method_norm not in tokens:
            continue
        if spec_norm and spec_norm not in tokens:
            continue

        yield path


def load_results_csvs(
    env: str,
    method: str | None = None,
    spec: str | None = None,
    results_dir: str | Path = "results",
) -> Dict[Path, pd.DataFrame]:
    """
    Load result CSVs for a given env with optional method/spec filters.

    Example:
        dfs = load_results_csvs("AcrobotSwingup", method="ppo", spec="brax")
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    matches = list(_iter_matching_csvs(results_dir, env, method, spec))
    return {path: pd.read_csv(path) for path in matches}


def _load_trials_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read {path} ({exc}).")
        return None
    if "steps" not in df.columns:
        print(f"Warning: missing 'steps' column in {path}.")
        return None
    trial_cols = [c for c in df.columns if c.startswith("trial_")]
    if not trial_cols:
        print(f"Warning: no trial_* columns in {path}.")
        return None
    df = df[["steps"] + trial_cols].dropna()
    if df.empty:
        return None
    return df


def _label_from_csv(path: Path, env_name: str) -> str:
    stem = path.stem
    if stem.startswith(env_name):
        remainder = stem[len(env_name) :].lstrip("_- ")
        return remainder or "default"
    return stem


def plot_results_csvs(
    env_names: list[str],
    method: str | None = None,
    spec: str | None = None,
    results_dir: str | Path = "results",
    out_dir: str | Path = os.path.join("results", "wandb_loader", "Figures", "csv"),
    max_steps: float | None = 5e7,
) -> int:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    any_plotted = False
    for env_name in env_names:
        matches = list(_iter_matching_csvs(results_dir, env_name, method, spec))
        if not matches:
            print(f"No CSVs found for {env_name}.")
            continue

        series = []
        for path in sorted(matches):
            df = _load_trials_csv(path)
            if df is None:
                continue
            label = _label_from_csv(path, env_name)
            series.append((label, df))

        if not series:
            print(f"No usable CSVs for {env_name}.")
            continue

        plt.figure(figsize=(9, 5))
        for label, df in series:
            trial_cols = [c for c in df.columns if c.startswith("trial_")]
            values = df[trial_cols].to_numpy()
            mean = values.mean(axis=1)
            n_trials = values.shape[1]
            if n_trials > 1:
                stderr = values.std(axis=1, ddof=1) / (n_trials**0.5)
            else:
                stderr = 0.0
            plt.plot(df["steps"], mean, label=f"{label} (n={n_trials})")
            plt.fill_between(
                df["steps"],
                mean - stderr,
                mean + stderr,
                alpha=0.2,
            )

        plt.xlabel("steps")
        plt.ylabel("return")
        plt.title(f"{env_name}: CSV results")
        plt.legend(loc="best", fontsize=8)
        if max_steps is not None:
            plt.xlim(left=0, right=max_steps)
        plt.tight_layout()

        output_path = out_dir / f"{env_name}_csv_results.png"
        plt.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")
        any_plotted = True

    return 0 if any_plotted else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot result CSVs by env, method, and spec."
    )
    parser.add_argument(
        "--method",
        default="ppo",
        help="Method token (default: ppo).",
    )
    parser.add_argument("--spec", default=None, help="Spec token, e.g. brax")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to results directory (default: results)",
    )
    parser.add_argument(
        "--envs",
        default=",".join(ENV_NAMES),
        help="Comma-separated env list for plotting.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("results", "wandb_loader", "Figures", "csv"),
        help="Output directory for plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    if not envs:
        envs = ENV_NAMES
    code = plot_results_csvs(
        env_names=envs,
        method="ppo",
        spec=args.spec,
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        max_steps=5e7,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
