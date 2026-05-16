"""Centralized Isaac/RSL-RL launcher for experiment-owned runtime specs.

This module stays lightweight at import time. Heavy Isaac/RSL-RL/gym/torch
imports happen only inside ``launch_from_runtime_spec`` after the runtime spec
environment variable has been set.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from utils.experiment.rl_runtime_spec import (
    RUNTIME_SPEC_ENV_VAR,
    load_runtime_spec_from_env,
)


def launch_from_runtime_spec(spec_path: str | Path) -> dict[str, Any]:
    """Launch RL training from a runtime spec written by ``scripts.train``."""

    path = Path(spec_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"RL runtime spec does not exist: {path}")
    os.environ[RUNTIME_SPEC_ENV_VAR] = str(path)
    spec = load_runtime_spec_from_env()
    os.environ["TOOL_GENERALIST_PATHS_YAML"] = str(Path(_require_string(spec, "paths_yaml")).expanduser())
    if _should_spawn_distributed(spec):
        return _spawn_distributed_workers(path, spec)
    return _launch_with_isaac(spec)


def _launch_with_isaac(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Run the IsaacLab/RSL-RL training flow.

    This function intentionally performs all heavyweight imports locally. Any
    failure after this point should be a real Isaac/RSL-RL import or API error,
    not an unfinished experiment-runner stub.
    """

    _log("before importing isaaclab.app.AppLauncher")
    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]]
    app_args = _build_app_launcher_args(AppLauncher, spec)
    _log(f"before AppLauncher task={spec.get('task_id')} args={vars(app_args)}")
    app_launcher = AppLauncher(app_args)
    _log("after AppLauncher creation")
    simulation_app = app_launcher.app
    launch_error: BaseException | None = None
    try:
        _log("before RSL-RL training flow")
        _run_rsl_rl_training(spec, app_launcher)
        _log("after RSL-RL training flow")
    except SystemExit as exc:
        launch_error = exc
        _log(f"SystemExit during Isaac/RSL-RL launch code={exc.code!r}")
        raise
    except BaseException as exc:
        launch_error = exc
        _log(f"exception during Isaac/RSL-RL launch type={type(exc).__name__} error={exc!r}")
        raise
    finally:
        if launch_error is None:
            _log("closing SimulationApp")
            simulation_app.close()
            _log("SimulationApp closed")
        else:
            _log(
                "skipping SimulationApp close after launch failure to avoid hiding "
                f"{type(launch_error).__name__}"
            )
    return {"launched": True, "task_id": spec["task_id"]}


def _should_spawn_distributed(spec: Mapping[str, Any]) -> bool:
    launch = _mapping(spec.get("launch_params"), "launch_params")
    return (
        bool(launch.get("distributed", False))
        and int(spec.get("num_gpus", 0)) > 1
        and "LOCAL_RANK" not in os.environ
    )


def _spawn_distributed_workers(spec_path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    num_gpus = int(spec["num_gpus"])
    env = os.environ.copy()
    env[RUNTIME_SPEC_ENV_VAR] = str(spec_path)
    env["TOOL_GENERALIST_PATHS_YAML"] = str(Path(_require_string(spec, "paths_yaml")).expanduser())
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node",
        str(num_gpus),
        "-m",
        "utils.experiment.isaac_rl_launcher",
        "--runtime-spec",
        str(spec_path),
    ]
    completed = subprocess.run(cmd, check=True, env=env)
    return {
        "launched": True,
        "distributed": True,
        "num_gpus": num_gpus,
        "returncode": completed.returncode,
        "task_id": spec["task_id"],
    }


def _build_app_launcher_args(app_launcher_cls: Any, spec: Mapping[str, Any]) -> argparse.Namespace:
    launch = _mapping(spec.get("launch_params"), "launch_params")
    parser = argparse.ArgumentParser(add_help=False)
    app_launcher_cls.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args([])

    _set_if_present(args, "headless", bool(launch.get("headless", True)))
    _set_if_present(args, "enable_cameras", bool(launch.get("enable_cameras", False)))
    _set_if_present(args, "disable_fabric", bool(launch.get("disable_fabric", False)))
    # AppLauncher does not add this argument itself; IsaacLab distributed setup
    # only runs when the key exists in the launcher config.
    setattr(args, "distributed", bool(launch.get("distributed", False)))
    if launch.get("device") is not None:
        _set_if_present(args, "device", str(launch["device"]))
    return args


def _run_rsl_rl_training(spec: Mapping[str, Any], app_launcher: Any) -> None:
    _log("before importing gymnasium/torch/IsaacLab/RSL-RL modules")
    import gymnasium as gym
    import torch
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    _log("after importing gymnasium/torch/IsaacLab/RSL-RL modules")

    _set_asset_assignment_rank_env(spec, app_launcher)

    _log("before importing isaaclab_tasks")
    import isaaclab_tasks  # noqa: F401
    _log("after importing isaaclab_tasks")
    _log("before importing IsaacLab_nonPrehensile.tasks")
    import IsaacLab_nonPrehensile.tasks  # noqa: F401
    _log("after importing IsaacLab_nonPrehensile.tasks")
    _log("before importing tool-sdf task registration module")
    import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile  # noqa: F401
    _log("after importing tool-sdf task registration module")

    task_id = _require_string(spec, "task_id")
    _ensure_gym_task_registered(gym, task_id)
    launch = _mapping(spec.get("launch_params"), "launch_params")
    artifact_dir = Path(_require_string(spec, "artifact_dir"))
    _log(
        "before hydra task config "
        f"task={task_id} num_envs={spec.get('num_envs')} "
        f"max_iterations={spec.get('max_iterations')} artifact={artifact_dir}"
    )

    @hydra_task_config(task_id, "rsl_rl_cfg_entry_point")
    def _main(env_cfg, agent_cfg):
        _log("entered hydra main")
        env_cfg.scene.num_envs = int(spec["num_envs"])
        if launch.get("device") is not None:
            env_cfg.sim.device = str(launch["device"])
            agent_cfg.device = str(launch["device"])
        seed = int(spec["seed"])
        env_cfg.seed = seed
        agent_cfg.seed = seed
        if bool(launch.get("distributed", False)):
            local_rank = _local_rank(app_launcher)
            env_cfg.sim.device = f"cuda:{local_rank}"
            agent_cfg.device = f"cuda:{local_rank}"
            seed = seed + local_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed
        if str(launch.get("logger", "tensorboard")) == "none":
            agent_cfg.logger = "tensorboard"
            log_dir = None
        else:
            agent_cfg.logger = str(launch.get("logger", "tensorboard"))
            if launch.get("wandb_project") and hasattr(agent_cfg, "wandb_project"):
                agent_cfg.wandb_project = str(launch["wandb_project"])
            if launch.get("run_name") and hasattr(agent_cfg, "run_name"):
                agent_cfg.run_name = str(launch["run_name"])
            if launch.get("wandb_project") and hasattr(agent_cfg, "experiment_name"):
                agent_cfg.experiment_name = str(launch["wandb_project"])
            log_dir = str(artifact_dir)

        _log(
            "before gym.make "
            f"task={task_id} env_device={getattr(env_cfg.sim, 'device', None)} "
            f"agent_device={agent_cfg.device} num_envs={env_cfg.scene.num_envs}"
        )
        env = gym.make(task_id, cfg=env_cfg, render_mode=None)
        _log("after gym.make")
        if isinstance(env.unwrapped, DirectMARLEnv):
            _log("before multi_agent_to_single_agent")
            env = multi_agent_to_single_agent(env)
            _log("after multi_agent_to_single_agent")
        _log("before RslRlVecEnvWrapper")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        _log("after RslRlVecEnvWrapper")

        _log(f"before OnPolicyRunner log_dir={log_dir} device={agent_cfg.device}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        _log("after OnPolicyRunner")
        try:
            _log(
                "before runner.learn "
                f"iterations={int(spec['max_iterations'])} "
                f"init_at_random_ep_len={bool(launch.get('init_at_random_ep_len', False))}"
            )
            runner.learn(
                num_learning_iterations=int(spec["max_iterations"]),
                init_at_random_ep_len=bool(launch.get("init_at_random_ep_len", False)),
            )
            _log("after runner.learn")
        except SystemExit as exc:
            _log(f"SystemExit inside hydra main code={exc.code!r}")
            raise
        finally:
            _log("closing RL env")
            env.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _log("RL env closed")

    _log("calling hydra main")
    try:
        _main()
    except SystemExit as exc:
        _log(f"SystemExit from hydra main code={exc.code!r}")
        raise
    except BaseException as exc:
        _log(f"exception from hydra main type={type(exc).__name__} error={exc!r}")
        raise
    _log("hydra main returned")


def _log(message: str) -> None:
    print(f"[rl_launcher] {message}", flush=True)


def _ensure_gym_task_registered(gym_module: Any, task_id: str) -> None:
    registry = getattr(getattr(gym_module, "envs", None), "registry", None)
    if registry is None:
        raise RuntimeError("Gymnasium registry is unavailable after task imports")
    if task_id in registry:
        _log(f"gym task registered task={task_id}")
        return
    available = sorted(str(key) for key in registry.keys() if "tool" in str(key).lower())
    raise RuntimeError(
        f"Gymnasium task {task_id!r} is not registered after importing "
        "IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile. "
        f"Available tool-like tasks: {available}"
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"RL runtime spec {name} must be an object")
    return value


def _require_string(spec: Mapping[str, Any], key: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"RL runtime spec requires non-empty {key}")
    return value


def _set_if_present(args: argparse.Namespace, name: str, value: Any) -> None:
    if hasattr(args, name):
        setattr(args, name, value)


def _set_asset_assignment_rank_env(spec: Mapping[str, Any], app_launcher: Any) -> None:
    """Expose rank metadata before task registration imports env_tool."""

    launch = _mapping(spec.get("launch_params"), "launch_params")
    if bool(launch.get("distributed", False)):
        global_rank = int(os.getenv("RANK", "0"))
        local_rank = _local_rank(app_launcher)
        world_size = int(os.getenv("WORLD_SIZE", str(max(1, int(spec.get("num_gpus", 1))))))
    else:
        global_rank = 0
        local_rank = 0
        world_size = 1

    os.environ["TOOL_GENERALIST_GLOBAL_RANK"] = str(global_rank)
    os.environ["TOOL_GENERALIST_LOCAL_RANK"] = str(local_rank)
    os.environ["TOOL_GENERALIST_WORLD_SIZE"] = str(world_size)


def _local_rank(app_launcher: Any) -> int:
    if hasattr(app_launcher, "local_rank"):
        return int(app_launcher.local_rank)
    return int(os.getenv("LOCAL_RANK", "0")) + int(os.getenv("JAX_LOCAL_RANK", "0"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Internal RL runtime-spec worker entrypoint.")
    parser.add_argument("--runtime-spec", required=True)
    args, _ = parser.parse_known_args(argv)
    launch_from_runtime_spec(args.runtime_spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
