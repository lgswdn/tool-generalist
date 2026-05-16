#!/usr/bin/env python3
"""Minimal Isaac Sim startup diagnostic.

Run this from the same environment used for contact generation.  It does not
import contact generation, torch, pretrain, RL, or project runtime code.  It
prefers IsaacLab's AppLauncher so the diagnostic uses the same headless
experience as RL training.
"""

from __future__ import annotations

import os
import sys
import time
import traceback


def _log(message: str) -> None:
    print(f"[debug_isaac_startup] {message}", flush=True)


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    return value if len(value) <= 240 else value[:240] + "..."


def _force_headless_argv() -> None:
    """Force Kit's no-window path before importing SimulationApp."""

    forced_args = (
        "--headless",
        "--no-window",
        "--/app/window/enabled=false",
        "--/app/viewport/enabled=false",
        "--/app/livestream/enabled=false",
    )
    for arg in forced_args:
        if arg not in sys.argv:
            sys.argv.append(arg)


def _headless_launch_config() -> dict:
    return {
        "headless": True,
        "hide_ui": True,
        "disable_viewport_updates": True,
        "enable_cameras": False,
        "width": 1,
        "height": 1,
        "window_width": 1,
        "window_height": 1,
        "display_options": 0,
    }


def _step_without_render() -> None:
    try:
        from isaacsim.core.api import World
    except Exception:
        try:
            from omni.isaac.core import World
        except Exception as exc:
            _log(
                "World API unavailable; skipping raw SimulationApp.update() "
                f"to avoid initializing RTX renderer in headless mode: {type(exc).__name__}: {exc}"
            )
            return

    _log("stepping World once with render=False")
    world = World(stage_units_in_meters=1.0)
    try:
        world.reset()
    except Exception as exc:
        _log(f"World.reset failed; continuing to step render=False: {type(exc).__name__}: {exc}")
    world.step(render=False)


def _start_app(started: float):
    try:
        _log("importing isaaclab.app.AppLauncher")
        from isaaclab.app import AppLauncher
    except Exception as exc:
        _log(
            "IsaacLab AppLauncher unavailable; falling back to raw SimulationApp "
            f"startup: {type(exc).__name__}: {exc}"
        )
    else:
        kit_args = " ".join(
            (
                "--/app/window/enabled=false",
                "--/app/viewport/enabled=false",
                "--/app/livestream/enabled=false",
            )
        )
        launcher_args = {
            "headless": True,
            "enable_cameras": False,
            "kit_args": kit_args,
        }
        _log(f"creating AppLauncher args={launcher_args}")
        launcher = AppLauncher(launcher_args)
        _log(f"AppLauncher app ready after {time.time() - started:.2f}s")
        return launcher.app, launcher

    _log("importing isaacsim.SimulationApp")
    from isaacsim import SimulationApp

    config = _headless_launch_config()
    _log(f"creating SimulationApp config={config}")
    app = SimulationApp(launch_config=config)
    _log(f"SimulationApp created after {time.time() - started:.2f}s")
    return app, None


def main() -> int:
    started = time.time()
    app = None
    launcher = None
    _force_headless_argv()
    _log(f"python={sys.executable}")
    _log(f"cwd={os.getcwd()}")
    _log(f"argv={sys.argv}")
    for key in (
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "DISPLAY",
        "VULKAN_SDK",
        "ISAACSIM_PATH",
        "EXP_PATH",
        "CARB_APP_PATH",
    ):
        _log(f"env {key}={_env(key)!r}")

    try:
        app, launcher = _start_app(started)

        _log("importing omni.usd and pxr.UsdGeom")
        import omni.usd
        from pxr import UsdGeom

        _log("creating blank stage")
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        _step_without_render()

        _log(f"startup diagnostic complete after {time.time() - started:.2f}s")
        return 0
    except BaseException:
        _log("startup diagnostic failed")
        traceback.print_exc()
        return 1
    finally:
        if app is not None:
            _log("closing SimulationApp")
            try:
                app.close()
            except BaseException:
                traceback.print_exc()
        launcher = None


if __name__ == "__main__":
    raise SystemExit(main())
