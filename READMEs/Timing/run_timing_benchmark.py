#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from timing_wandb_utils import (
    canonical_method,
    parse_method_keyvals,
    parse_method_overrides,
    split_csv,
)


METHOD_SPECS: dict[str, dict[str, Any]] = {
    "reppo": {"module": "src.jaxrl.reppo", "supports_diffusion_steps": False},
    "ppo": {"module": "src.jaxrl.reppo_PPO", "supports_diffusion_steps": False},
    "dime": {"module": "src.jaxrl.reppo_dime", "supports_diffusion_steps": True},
    "diffppo": {"module": "src.jaxrl.DA_MDP_PPO", "supports_diffusion_steps": True},
    "dmerl": {"module": "src.jaxrl.DA_MDP_REPPO", "supports_diffusion_steps": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch timing benchmark runs across methods and env names with "
            "per-method Hydra overrides."
        )
    )
    parser.add_argument(
        "--methods",
        default="reppo,ppo,dime,diffppo,dmerl",
        help="Comma-separated methods. Supported: reppo, ppo, dime, diffppo, dmerl.",
    )
    parser.add_argument(
        "--env-names",
        required=True,
        help="Comma-separated env.name values. Example: WalkerRun,HopperHop",
    )
    parser.add_argument(
        "--default-env",
        default="mjx_dmc",
        help="Fallback Hydra env=... value if method-specific value is not set.",
    )
    parser.add_argument(
        "--method-env",
        action="append",
        default=[],
        help="Per-method Hydra env=... override, format method:env_cfg (repeatable).",
    )
    parser.add_argument(
        "--default-experiment-overrides",
        default="default",
        help="Fallback experiment_overrides=... value if method-specific value is not set.",
    )
    parser.add_argument(
        "--method-experiment-overrides",
        action="append",
        default=[],
        help=(
            "Per-method experiment_overrides=... value, format "
            "method:override_name (repeatable)."
        ),
    )
    parser.add_argument(
        "--method-diff-steps",
        action="append",
        default=[],
        help=(
            "Per-method diffusion steps, format method:int. "
            "Used as hyperparameters.diffusion.diff_steps=... (repeatable)."
        ),
    )
    parser.add_argument(
        "--total-time-steps",
        type=int,
        default=None,
        help="If set, adds hyperparameters.total_time_steps=<value> to every run.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument(
        "--num-seeds", type=int, default=None, help="Optional num_seeds override."
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Optional trials/num_trials override depending on method.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Global Hydra override key=value (repeatable).",
    )
    parser.add_argument(
        "--method-extra-override",
        action="append",
        default=[],
        help="Per-method extra override in format method:key=value (repeatable).",
    )
    parser.add_argument(
        "--project-suffix",
        default="_timing",
        help=(
            "W&B project suffix injected into each run as "
            "wandb.project_suffix=<value>. Set to empty string to disable."
        ),
    )
    parser.add_argument(
        "--python-exec",
        default=sys.executable,
        help="Python executable used to launch runs.",
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="Working directory used for subprocesses (usually repo root).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if a command fails.",
    )
    return parser.parse_args()


def parse_method_ints(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(
                f"Expected 'method:int' format, got '{item}'."
            )
        method, value = item.split(":", 1)
        out[canonical_method(method)] = int(value.strip())
    return out


def build_command(
    *,
    method: str,
    env_name: str,
    method_env_cfg: str,
    experiment_overrides: str,
    method_diff_steps: int | None,
    args: argparse.Namespace,
    method_overrides: dict[str, list[str]],
) -> list[str]:
    spec = METHOD_SPECS[method]
    cmd = [
        args.python_exec,
        "-m",
        spec["module"],
        f"env={method_env_cfg}",
        f"env.name={env_name}",
        f"experiment_overrides={experiment_overrides}",
    ]

    if args.total_time_steps is not None:
        cmd.append(f"hyperparameters.total_time_steps={args.total_time_steps}")
    if args.seed is not None:
        cmd.append(f"seed={args.seed}")
    if args.num_seeds is not None:
        cmd.append(f"num_seeds={args.num_seeds}")
    if args.num_trials is not None:
        if method in {"ppo", "diffppo"}:
            cmd.append(f"trials={args.num_trials}")
        else:
            cmd.append(f"num_trials={args.num_trials}")
    if method_diff_steps is not None and spec["supports_diffusion_steps"]:
        cmd.append(f"hyperparameters.diffusion.diff_steps={method_diff_steps}")

    extra_overrides = [*args.extra_override, *method_overrides.get(method, [])]
    has_project_suffix_override = any(
        override.startswith("wandb.project_suffix=") for override in extra_overrides
    )
    if args.project_suffix and not has_project_suffix_override:
        cmd.append(f"wandb.project_suffix={args.project_suffix}")

    cmd.extend(extra_overrides)
    return cmd


def main() -> int:
    args = parse_args()
    methods = [canonical_method(m) for m in split_csv(args.methods)]
    env_names = split_csv(args.env_names)
    if not env_names:
        raise ValueError("No env names provided.")

    unknown = [m for m in methods if m not in METHOD_SPECS]
    if unknown:
        raise ValueError(f"Unknown method ids: {unknown}. Supported: {sorted(METHOD_SPECS)}")

    method_env = parse_method_keyvals(args.method_env)
    method_exp = parse_method_keyvals(args.method_experiment_overrides)
    method_diff_steps = parse_method_ints(args.method_diff_steps)
    method_overrides = parse_method_overrides(args.method_extra_override)

    workdir = Path(args.workdir).resolve()
    failures: list[tuple[str, int]] = []
    launched = 0

    print(f"[info] workdir={workdir}")
    print(f"[info] methods={methods}")
    print(f"[info] env_names={env_names}")

    for env_name in env_names:
        for method in methods:
            env_cfg = method_env.get(method, args.default_env)
            exp_cfg = method_exp.get(method, args.default_experiment_overrides)
            diff_steps = method_diff_steps.get(method)

            cmd = build_command(
                method=method,
                env_name=env_name,
                method_env_cfg=env_cfg,
                experiment_overrides=exp_cfg,
                method_diff_steps=diff_steps,
                args=args,
                method_overrides=method_overrides,
            )
            launched += 1
            cmd_text = " ".join(shlex.quote(part) for part in cmd)
            print(f"[launch {launched}] {cmd_text}")

            if args.dry_run:
                continue

            result = subprocess.run(cmd, cwd=workdir, check=False)
            if result.returncode != 0:
                failures.append((f"{method}:{env_name}", result.returncode))
                if args.stop_on_error:
                    print(f"[error] command failed with exit code {result.returncode}")
                    return result.returncode

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    if failures:
        print("[summary] some runs failed:")
        for key, code in failures:
            print(f"  - {key}: exit_code={code}")
        return 1

    print(f"[summary] completed {launched} run launches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
