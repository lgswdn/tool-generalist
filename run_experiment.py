#!/usr/bin/env python3
"""Run or plan an experiment from an ExpCfg."""

from __future__ import annotations

import argparse

from utils.experiment.runner import format_summary, plan_from_config, run_from_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Python file or module exposing EXP_CFG")
    parser.add_argument("--mode", choices=("run", "plan"), default="run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = (
        plan_from_config(args.config)
        if args.mode == "plan"
        else run_from_config(args.config)
    )
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
