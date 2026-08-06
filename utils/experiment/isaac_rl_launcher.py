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
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
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

    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]]
    app_args = _build_app_launcher_args(AppLauncher, spec)
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app
    launch_error: BaseException | None = None
    try:
        _run_rsl_rl_training(spec, app_launcher)
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
            simulation_app.close()
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
    env.setdefault("HYDRA_FULL_ERROR", "1")
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
    import gymnasium as gym
    import torch
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    _set_asset_assignment_rank_env(spec, app_launcher)

    import isaaclab_tasks  # noqa: F401
    import IsaacLab_nonPrehensile.tasks  # noqa: F401
    import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile  # noqa: F401

    task_id = _require_string(spec, "task_id")
    _ensure_gym_task_registered(gym, task_id)
    launch = _mapping(spec.get("launch_params"), "launch_params")
    artifact_dir = Path(_require_string(spec, "artifact_dir"))
    _log(
        f"start task={task_id} num_envs={spec.get('num_envs')} "
        f"iterations={spec.get('max_iterations')} artifact={artifact_dir}"
    )

    @hydra_task_config(task_id, "rsl_rl_cfg_entry_point")
    def _main(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = int(spec["num_envs"])
        disabled_markers = _disable_training_visualization_markers(env_cfg)
        if disabled_markers:
            _log(f"disabled training visualization markers={','.join(disabled_markers)}")
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
            if hasattr(agent_cfg, "wandb_upload_files"):
                agent_cfg.wandb_upload_files = bool(launch.get("wandb_upload_files", False))
            if launch.get("run_name") and hasattr(agent_cfg, "run_name"):
                agent_cfg.run_name = str(launch["run_name"])
            if launch.get("wandb_project") and hasattr(agent_cfg, "experiment_name"):
                agent_cfg.experiment_name = str(launch["wandb_project"])
            log_dir = str(artifact_dir)

        try:
            env = gym.make(task_id, cfg=env_cfg, render_mode=None)
        except TypeError as exc:
            raise _concise_gym_make_error(task_id, exc) from (exc.__context__ or exc)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        init_checkpoint = spec.get("rl_init_checkpoint")
        if isinstance(init_checkpoint, str) and init_checkpoint.strip():
            init_path = Path(init_checkpoint).expanduser().resolve()
            if not init_path.is_file():
                raise FileNotFoundError(f"RL initialization checkpoint does not exist: {init_path}")
            _log(f"initialize policy weights checkpoint={init_path}")
            loaded = torch.load(init_path, map_location=agent_cfg.device, weights_only=False)
            if not isinstance(loaded, dict) or not isinstance(loaded.get("model_state_dict"), dict):
                raise RuntimeError(
                    f"RL initialization checkpoint must contain model_state_dict: {init_path}"
                )
            runner.alg.policy.load_state_dict(loaded["model_state_dict"], strict=True)
        resume_checkpoint = spec.get("rl_resume_checkpoint")
        if isinstance(resume_checkpoint, str) and resume_checkpoint.strip():
            resume_path = Path(resume_checkpoint).expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"RL resume checkpoint does not exist: {resume_path}")
            _log(f"resume checkpoint={resume_path}")
            runner.load(str(resume_path), load_optimizer=True)
        try:
            _log(f"learn iterations={int(spec['max_iterations'])}")
            runner.learn(
                num_learning_iterations=int(spec["max_iterations"]),
                init_at_random_ep_len=bool(launch.get("init_at_random_ep_len", False)),
                print_fine_grained_timing=bool(
                    launch.get("print_fine_grained_timing", False)
                ),
            )
            _log("learn complete")
        except SystemExit as exc:
            _log(f"SystemExit inside hydra main code={exc.code!r}")
            raise
        finally:
            env.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    try:
        _main()
    except SystemExit as exc:
        _log(f"SystemExit from hydra main code={exc.code!r}")
        raise
    except BaseException as exc:
        _log(f"exception from hydra main type={type(exc).__name__} error={exc!r}")
        raise


def _log(message: str) -> None:
    print(f"[rl_launcher] {message}", flush=True)


def _concise_gym_make_error(task_id: str, exc: TypeError) -> RuntimeError:
    """Convert Gymnasium's huge env_creator kwargs dump into the root error."""

    root = exc.__context__ or exc.__cause__
    if root is not None:
        message = f"{type(root).__name__}: {_truncate_exception_message(root)}"
    else:
        message = f"{type(exc).__name__}: {_truncate_exception_message(exc)}"
    _log(f"gym.make failed task={task_id} root={message}")
    return RuntimeError(f"gym.make failed for task {task_id}: {message}")


def _truncate_exception_message(exc: BaseException, *, limit: int = 1200) -> str:
    text = str(exc)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ... [truncated]"


def _disable_training_visualization_markers(env_cfg: Any) -> list[str]:
    """Disable marker drawing for training without changing video preferences.

    Environment ``visualize_*`` values and command ``debug_vis`` values remain
    the source of truth for interactive use and video recording.  The training
    launcher overrides only the instantiated config passed to ``gym.make``.
    """

    disabled: list[str] = []
    for name in dir(env_cfg):
        if not name.startswith("visualize_"):
            continue
        try:
            value = getattr(env_cfg, name)
        except Exception:
            continue
        if isinstance(value, bool):
            setattr(env_cfg, name, False)
            if value:
                disabled.append(name)

    commands_cfg = getattr(env_cfg, "commands", None)
    if commands_cfg is not None:
        for term_name in dir(commands_cfg):
            if term_name.startswith("_"):
                continue
            try:
                term_cfg = getattr(commands_cfg, term_name)
                debug_vis = getattr(term_cfg, "debug_vis", None)
            except Exception:
                continue
            if isinstance(debug_vis, bool):
                term_cfg.debug_vis = False
                if debug_vis:
                    disabled.append(f"commands.{term_name}.debug_vis")

    return disabled


def _ensure_gym_task_registered(gym_module: Any, task_id: str) -> None:
    registry = getattr(getattr(gym_module, "envs", None), "registry", None)
    if registry is None:
        raise RuntimeError("Gymnasium registry is unavailable after task imports")
    if task_id in registry:
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
