#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd
import wandb


TIMING_KEYS = (
    "timing/rollout_seconds",
    "timing/update_seconds",
    "timing/total_seconds",
)
SPS_KEYS = ("sps", "system/sps")

METHOD_DISPLAY = {
    "reppo": "REPPO",
    "dime": "REPPO-DiME",
    "dmerl": "DME-REPPO",
    "ppo": "PPO",
    "diffppo": "DiffPPO",
}

METHOD_ORDER = ["reppo", "ppo", "dime", "diffppo", "dmerl"]

_METHOD_ALIASES = {
    "reppo": "reppo",
    "reppo.py": "reppo",
    "reppo_dime": "dime",
    "dime": "dime",
    "reppodime": "dime",
    "reppo_dmerl_new": "dmerl",
    "dmerl": "dmerl",
    "dmereppo": "dmerl",
    "dme_reppo": "dmerl",
    "reppo_ppo": "ppo",
    "ppo": "ppo",
    "reppo_diffppo": "diffppo",
    "diffppo": "diffppo",
    "diff_ppo": "diffppo",
    "dme_ppo": "diffppo",
}


@dataclass
class CollectedTiming:
    series_df: pd.DataFrame
    run_summary_df: pd.DataFrame


def split_csv(text: str | None) -> list[str]:
    if not text:
        return []
    out = []
    for token in text.split(","):
        clean = token.strip()
        if clean:
            out.append(clean)
    return out


def parse_method_keyvals(items: Sequence[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not items:
        return parsed
    for item in items:
        if ":" not in item:
            raise ValueError(
                f"Expected 'method:value' format, got '{item}'."
            )
        method, value = item.split(":", 1)
        parsed[canonical_method(method)] = value.strip()
    return parsed


def parse_method_overrides(items: Sequence[str] | None) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    if not items:
        return parsed
    for item in items:
        if ":" not in item:
            raise ValueError(
                f"Expected 'method:key=value' format, got '{item}'."
            )
        method, override = item.split(":", 1)
        method_key = canonical_method(method)
        parsed.setdefault(method_key, []).append(override.strip())
    return parsed


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def canonical_method(raw: str | None) -> str:
    if not raw:
        return "unknown"
    norm = _normalize_token(raw)
    if norm in _METHOD_ALIASES:
        return _METHOD_ALIASES[norm]
    return raw.lower().strip()


def method_display(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


def nested_get(data: dict[str, Any] | None, dotted_key: str) -> Any:
    if not isinstance(data, dict):
        return None
    if dotted_key in data:
        return data[dotted_key]
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def infer_method(run: wandb.apis.public.Run) -> str:
    cfg = dict(run.config or {})
    for key in ("method_name", "method", "name"):
        val = nested_get(cfg, key)
        method = canonical_method(str(val) if val is not None else None)
        if method in METHOD_DISPLAY:
            return method

    name_guess = canonical_method(run.name)
    if name_guess in METHOD_DISPLAY:
        return name_guess
    return "unknown"


def infer_env_name(run: wandb.apis.public.Run, project_name: str | None = None) -> str:
    cfg = dict(run.config or {})
    env_name = nested_get(cfg, "env.name")
    if isinstance(env_name, str) and env_name:
        return env_name
    if project_name and project_name.startswith("dime_"):
        return project_name.split("dime_", 1)[1]
    return "unknown_env"


def resolve_projects(projects: Sequence[str] | None, env_names: Sequence[str], prefix: str) -> list[str]:
    if projects:
        return list(projects)
    return [f"{prefix}{env_name}" for env_name in env_names]


def load_timing_series(
    run: wandb.apis.public.Run,
    drop_first: int,
    timing_keys: Sequence[str] = TIMING_KEYS,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in run.scan_history(keys=["_step", *SPS_KEYS, *timing_keys]):
        step = row.get("_step")
        entry: dict[str, float] = {"_step": float(step) if step is not None else np.nan}
        has_finite = False
        sps_val = np.nan
        for sps_key in SPS_KEYS:
            candidate = row.get(sps_key, np.nan)
            try:
                candidate_float = float(candidate)
            except Exception:
                candidate_float = np.nan
            if math.isfinite(candidate_float):
                sps_val = candidate_float
                break
        try:
            sps_float = float(sps_val)
        except Exception:
            sps_float = np.nan
        if math.isfinite(sps_float):
            has_finite = True
        entry["sps"] = sps_float
        for key in timing_keys:
            val = row.get(key, np.nan)
            try:
                val_float = float(val)
            except Exception:
                val_float = np.nan
            if math.isfinite(val_float):
                has_finite = True
            entry[key] = val_float
        if has_finite:
            rows.append(entry)

    if not rows:
        return pd.DataFrame(columns=["_step", "sps", *timing_keys])

    df = pd.DataFrame(rows).sort_values("_step").reset_index(drop=True)
    if "timing/total_seconds" in df.columns and "sps" in df.columns:
        # For SPS-only logging runs, estimate total wall-time per interval:
        # delta_seconds ~= delta_env_steps / sps.
        has_total = pd.to_numeric(df["timing/total_seconds"], errors="coerce").notna().any()
        if not has_total:
            delta_steps = pd.to_numeric(df["_step"], errors="coerce").diff()
            sps_series = pd.to_numeric(df["sps"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                est_total = delta_steps / sps_series
            est_total = est_total.where((est_total >= 0.0) & np.isfinite(est_total), np.nan)
            df["timing/total_seconds"] = est_total

    if drop_first > 0:
        df = df.iloc[drop_first:].reset_index(drop=True)
    return df


def summarize_run(df: pd.DataFrame, timing_keys: Sequence[str] = TIMING_KEYS) -> dict[str, float]:
    summary: dict[str, float] = {"samples": float(len(df))}
    sps_series = pd.to_numeric(df.get("sps", pd.Series(dtype=float)), errors="coerce").dropna()
    if sps_series.empty:
        summary["sps/mean"] = np.nan
        summary["sps/median"] = np.nan
        summary["sps/std"] = np.nan
    else:
        summary["sps/mean"] = float(sps_series.mean())
        summary["sps/median"] = float(sps_series.median())
        summary["sps/std"] = float(sps_series.std(ddof=0))
    for key in timing_keys:
        series = pd.to_numeric(df.get(key, pd.Series(dtype=float)), errors="coerce").dropna()
        if series.empty:
            summary[f"{key}/mean"] = np.nan
            summary[f"{key}/median"] = np.nan
            summary[f"{key}/std"] = np.nan
            summary[f"{key}/integrated"] = np.nan
        else:
            summary[f"{key}/mean"] = float(series.mean())
            summary[f"{key}/median"] = float(series.median())
            summary[f"{key}/std"] = float(series.std(ddof=0))
            summary[f"{key}/integrated"] = float(series.sum())
    return summary


def collect_timing_data(
    *,
    entity: str,
    projects: Sequence[str],
    allowed_methods: set[str] | None,
    state: str | None,
    drop_first: int,
    min_points: int,
    verbose: bool = False,
) -> CollectedTiming:
    api = wandb.Api()
    series_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for project in projects:
        path = f"{entity}/{project}" if entity else project
        filters = {"state": state} if state and state != "any" else None
        try:
            runs = api.runs(path=path, filters=filters)
        except Exception as exc:
            if verbose:
                print(f"[warn] failed to list runs for {path}: {exc}")
            continue

        try:
            for run in runs:
                method = infer_method(run)
                if allowed_methods and method not in allowed_methods:
                    continue
                env_name = infer_env_name(run, project_name=project)
                timing_df = load_timing_series(run, drop_first=drop_first)
                if len(timing_df) < min_points:
                    if verbose:
                        print(
                            f"[skip] {run.id} ({run.name}) env={env_name} method={method} "
                            f"has only {len(timing_df)} timing points after drop_first={drop_first}."
                        )
                    continue

                run_summary = summarize_run(timing_df)
                run_rows.append(
                    {
                        "project": project,
                        "run_id": run.id,
                        "run_name": run.name,
                        "state": run.state,
                        "env_name": env_name,
                        "method": method,
                        "method_display": method_display(method),
                        **run_summary,
                    }
                )

                for _, row in timing_df.iterrows():
                    series_rows.append(
                        {
                            "project": project,
                            "run_id": run.id,
                            "run_name": run.name,
                            "env_name": env_name,
                            "method": method,
                            "method_display": method_display(method),
                            "_step": row["_step"],
                            **{key: row.get(key, np.nan) for key in TIMING_KEYS},
                        }
                    )
        except Exception as exc:
            if verbose:
                print(f"[warn] failed to iterate runs for {path}: {exc}")
            continue

    series_df = pd.DataFrame(series_rows)
    run_summary_df = pd.DataFrame(run_rows)
    return CollectedTiming(series_df=series_df, run_summary_df=run_summary_df)


def aggregate_env_method(run_summary_df: pd.DataFrame) -> pd.DataFrame:
    if run_summary_df.empty:
        return pd.DataFrame()

    grouped = (
        run_summary_df
        .groupby(["env_name", "method", "method_display"], dropna=False)
        .agg(
            runs=("run_id", "nunique"),
            sps_mean=("sps/mean", "mean"),
            sps_std=("sps/mean", "std"),
            rollout_mean=("timing/rollout_seconds/mean", "mean"),
            rollout_std=("timing/rollout_seconds/mean", "std"),
            update_mean=("timing/update_seconds/mean", "mean"),
            update_std=("timing/update_seconds/mean", "std"),
            total_mean=("timing/total_seconds/mean", "mean"),
            total_std=("timing/total_seconds/mean", "std"),
            rollout_integrated_mean=("timing/rollout_seconds/integrated", "mean"),
            rollout_integrated_std=("timing/rollout_seconds/integrated", "std"),
            update_integrated_mean=("timing/update_seconds/integrated", "mean"),
            update_integrated_std=("timing/update_seconds/integrated", "std"),
            total_integrated_mean=("timing/total_seconds/integrated", "mean"),
            total_integrated_std=("timing/total_seconds/integrated", "std"),
        )
        .reset_index()
    )
    grouped["method_order"] = grouped["method"].map(
        {name: idx for idx, name in enumerate(METHOD_ORDER)}
    ).fillna(1e9)
    grouped = grouped.sort_values(["env_name", "method_order", "method"]).drop(
        columns=["method_order"]
    )
    return grouped
