#!/usr/bin/env python3
"""Run or plan an experiment from an ExpCfg."""

from __future__ import annotations

import argparse

from utils.experiment.runner import format_summary, plan_from_config, run_from_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Python file or module exposing EXP_CFG")
    parser.add_argument("--mode", choices=("run", "plan"), default="run")
    parser.add_argument(
        "--curriculum-from-eval",
        action="store_true",
        help=(
            "For RL training, derive a runtime-only object manifest from the latest "
            "eval_objects_summary.json for this experiment, selecting objects with "
            "success_rate <= --curriculum-success-rate-threshold."
        ),
    )
    parser.add_argument(
        "--curriculum-success-rate-threshold",
        type=float,
        default=0.2,
        help="Success-rate threshold used by --curriculum-from-eval.",
    )
    parser.add_argument(
        "--no-curriculum-resume",
        action="store_true",
        help="With --curriculum-from-eval, do not resume RL weights from the evaluated checkpoint.",
    )
    parser.add_argument(
        "--runtime-num-gpus",
        type=int,
        default=None,
        help="Runtime-only RL GPU count override. Does not change experiment artifact hashes.",
    )
    parser.add_argument(
        "--runtime-total-envs",
        type=int,
        default=8192,
        help="Total RL env count to preserve when --runtime-num-gpus is set.",
    )
    parser.add_argument(
        "--print-fine-grained-timing",
        action="store_true",
        help=(
            "Print the existing per-iteration RL collect timing breakdown "
            "(encoder, actor_critic, and other)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "plan":
        result = plan_from_config(args.config)
    else:
        result = run_from_config(
            args.config,
            curriculum_from_eval=args.curriculum_from_eval,
            curriculum_success_rate_threshold=args.curriculum_success_rate_threshold,
            curriculum_resume_from_eval=not args.no_curriculum_resume,
            runtime_num_gpus=args.runtime_num_gpus,
            runtime_total_envs=args.runtime_total_envs,
            runtime_print_fine_grained_timing=args.print_fine_grained_timing,
        )
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
